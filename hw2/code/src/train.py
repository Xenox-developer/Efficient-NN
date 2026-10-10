import argparse
import csv
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from .models import CifarResNet
from .data import datasets, loader
from .schedules import schedule


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for x, y in batches:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return loss_sum / count, correct / count


def metadata(config, experiment, seed, device, synthetic):
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    try:
        torchvision_version = version('torchvision')
    except PackageNotFoundError:
        torchvision_version = None
    return dict(config=config, experiment=experiment, seed=seed, synthetic=synthetic,
                source_sha256=source_hash.hexdigest(), python=platform.python_version(),
                torch=torch.__version__, torchvision=torchvision_version, numpy=np.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                precision='FP32', split_seed=2026, test_selection='best validation accuracy',
                train_size=64 if synthetic else 45000, val_size=16 if synthetic else 5000,
                test_size=16 if synthetic else 10000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/kaggle.json')
    ap.add_argument('--experiment', default='onecycle')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--output', default='results')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--stop-after-epoch', type=int, help='save a partial run for later resume')
    ap.add_argument('--synthetic', action='store_true', help='software smoke test only')
    args = ap.parse_args()
    all_cfg = json.loads(Path(args.config).read_text())
    exp = all_cfg['lr_range'] if args.experiment == 'lr_range' else all_cfg['experiments'][args.experiment]
    config = {k: v for k, v in all_cfg.items() if k not in ('experiments', 'lr_range')}
    config.update(exp)
    if args.synthetic:
        config.update(depth=8, batch_size=16, num_workers=0)
        if args.experiment == 'lr_range':
            config.update(steps=8, validate_every=2)
        else:
            config['epochs'] = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    seed_all(args.seed)
    out = Path(args.output) / f'{args.experiment}_seed{args.seed}'

    if args.synthetic and 'results' in Path(args.output).parts:
        raise ValueError('choose a separate --output directory for synthetic smoke tests')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError(f'{out} already completed. Choose a new output path.')
    if any(out.iterdir()) and not args.resume:
        raise FileExistsError(f'{out} exists. Use --resume or a new output path.')
    train_ds, val_ds, test_ds = datasets(args.data_root, args.synthetic)
    batches_per_epoch = math.ceil(len(train_ds) / config['batch_size'])
    total_steps = config['steps'] if args.experiment == 'lr_range' else config['epochs'] * batches_per_epoch
    val_loader = loader(val_ds, config['batch_size'], config['num_workers'], 2026, False, device.type == 'cuda')
    model = CifarResNet(config['depth'], config['bn_momentum']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=schedule(config, 0, total_steps)[0],
                                momentum=0.9, weight_decay=config['weight_decay'])
    meta = metadata(config, args.experiment, args.seed, device, args.synthetic)
    meta['parameter_count'] = sum(x.numel() for x in model.parameters())
    meta['steps_per_epoch'] = batches_per_epoch
    if args.experiment == 'lr_range':
        if args.resume:
            raise ValueError('LR range test has no resume. Choose a fresh output path.')
        atomic_json(out / 'metadata.json', meta)
        records, step, smooth, best, diverged = [], 0, 0.0, float('inf'), False
        sync(device)
        range_started = time.perf_counter()
        while step < total_steps and not diverged:
            train_loader = loader(train_ds, config['batch_size'], config['num_workers'],
                                  args.seed + step // batches_per_epoch, True, device.type == 'cuda')
            for x, y in train_loader:
                lr, momentum = schedule(config, step, total_steps)
                for group in optimizer.param_groups:
                    group.update(lr=lr, momentum=momentum)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
                raw = loss.item()
                if not math.isfinite(raw):
                    diverged = True
                    break
                loss.backward()
                optimizer.step()
                smooth = 0.98 * smooth + 0.02 * raw
                corrected = smooth / (1 - 0.98 ** (step + 1))
                best = min(best, corrected)
                row = dict(step=step + 1, lr=lr, loss=raw, smooth_loss=corrected, val_loss='', val_accuracy='')
                if (step + 1) % config['validate_every'] == 0 or step + 1 == total_steps:
                    row['val_loss'], row['val_accuracy'] = evaluate(model, val_loader, device)
                    print(f'LR range {step+1}/{total_steps}, lr={lr:.4g}, val={row["val_accuracy"]:.3f}', flush=True)
                records.append(row)
                step += 1
                if step >= 20 and corrected > 4 * best:
                    diverged = True
                if step >= total_steps or diverged:
                    break
        write_csv(out / 'lr_range.csv', records)
        sync(device)
        atomic_json(out / 'summary.json', dict(experiment=args.experiment, seed=args.seed,
                    synthetic=args.synthetic, updates=step, diverged=diverged, test_evaluated=False,
                    elapsed_seconds=time.perf_counter() - range_started))
        return
    rows, best_accuracy, best_epoch = [], -1.0, 0
    train_seconds, total_seconds, start_epoch = 0.0, 0.0, 0
    checkpoint_path = out / 'last.pt'
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        old_meta = json.loads((out / 'metadata.json').read_text())
        for field in ('config', 'seed', 'source_sha256', 'synthetic', 'experiment', 'torch', 'torchvision', 'gpu'):
            if old_meta[field] != meta[field]:
                raise ValueError(f'resume mismatch: {field}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        rows, start_epoch = checkpoint['rows'], checkpoint['epoch']
        best_accuracy, best_epoch = checkpoint['best_accuracy'], checkpoint['best_epoch']
        train_seconds, total_seconds = checkpoint['train_seconds'], checkpoint['total_seconds']
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        random.setstate(checkpoint['random_rng'])
        np.random.set_state(checkpoint['numpy_rng'])

        write_csv(out / 'history.csv', rows)
    else:
        atomic_json(out / 'metadata.json', meta)
    for epoch in range(start_epoch, config['epochs']):

        seed_all(args.seed + epoch)
        batches = loader(train_ds, config['batch_size'], config['num_workers'], args.seed + epoch,
                         True, device.type == 'cuda')
        model.train()
        sync(device)
        epoch_started = time.perf_counter()
        loss_sum, correct, count = 0.0, 0, 0
        for batch, (x, y) in enumerate(batches):
            step = epoch * batches_per_epoch + batch
            lr, momentum = schedule(config, step, total_steps)
            for group in optimizer.param_groups:
                group.update(lr=lr, momentum=momentum)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if not math.isfinite(loss.item()):
                atomic_json(out / 'failure.json', dict(epoch=epoch + 1, step=step + 1, lr=lr, reason='nonfinite loss'))
                raise RuntimeError(f'divergence at update {step+1}, lr={lr}')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.detach().argmax(1) == y).sum().item()
            count += len(y)
        sync(device)
        train_seconds += time.perf_counter() - epoch_started
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        sync(device)
        total_seconds += time.perf_counter() - epoch_started
        row = dict(epoch=epoch + 1, step=(epoch + 1) * batches_per_epoch,
                   lr=lr, momentum=momentum, train_loss=loss_sum / count,
                   train_accuracy=correct / count, val_loss=val_loss, val_accuracy=val_accuracy,
                   train_seconds=train_seconds, elapsed_seconds=total_seconds)
        rows.append(row)

        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch + 1
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch == start_epoch and args.resume:
            best_model = checkpoint['best_model']
        checkpoint = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          best_model=best_model, best_accuracy=best_accuracy, best_epoch=best_epoch,
                          rows=rows, train_seconds=train_seconds, total_seconds=total_seconds,
                          torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                          random_rng=random.getstate(), numpy_rng=np.random.get_state())
        tmp = out / 'last.tmp'
        torch.save(checkpoint, tmp)
        os.replace(tmp, checkpoint_path)
        write_csv(out / 'history.csv', rows)
        print(f'{args.experiment} seed={args.seed} epoch={epoch+1}/{config["epochs"]} '
              f'val={val_accuracy:.4f} time={total_seconds:.1f}s', flush=True)
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch and epoch + 1 < config['epochs']:
            print('Partial run saved. Continue with --resume.', flush=True)
            return

    if start_epoch == config['epochs']:
        best_model = checkpoint['best_model']
    final_val_accuracy = rows[-1]['val_accuracy']
    model.load_state_dict(best_model)
    test_loader = loader(test_ds, config['batch_size'], config['num_workers'], 2026, False, device.type == 'cuda')
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    reached = next((r for r in rows if r['val_accuracy'] >= config['target_val_accuracy']), None)
    atomic_json(out / 'summary.json', dict(experiment=args.experiment, seed=args.seed, synthetic=args.synthetic,
                epochs=config['epochs'], updates=total_steps, final_val_accuracy=final_val_accuracy,
                best_val_accuracy=best_accuracy, selected_epoch=best_epoch,
                test_accuracy=test_accuracy, test_loss=test_loss,
                train_seconds=train_seconds, elapsed_seconds=total_seconds,
                target_val_accuracy=config['target_val_accuracy'],
                steps_to_target=reached['step'] if reached else None,
                seconds_to_target=reached['elapsed_seconds'] if reached else None))


if __name__ == '__main__':
    main()
