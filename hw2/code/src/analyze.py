import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    args = ap.parse_args()
    root = Path(args.results)
    runs = []
    for path in sorted(root.glob('*/summary.json')):
        summary = json.loads(path.read_text())
        if summary.get('synthetic'):
            raise ValueError('Synthetic smoke results cannot be used in a reproduction report.')
        if summary['experiment'] == 'lr_range':
            continue
        meta = json.loads((path.parent / 'metadata.json').read_text())
        runs.append((path.parent, summary, meta))
    figures = root / 'figures'
    figures.mkdir(parents=True, exist_ok=True)

    for path in root.glob('lr_range_seed*/lr_range.csv'):
        with path.open() as f:
            records = list(csv.DictReader(f))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot([float(r['lr']) for r in records], [float(r['smooth_loss']) for r in records])
        axes[0].set(xlabel='Learning rate', ylabel='Smoothed training loss', title='LR range test')
        val = [r for r in records if r['val_accuracy']]
        axes[1].plot([float(r['lr']) for r in val], [100*float(r['val_accuracy']) for r in val], 'o-')
        axes[1].set(xlabel='Learning rate', ylabel='Validation accuracy (%)')
        fig.tight_layout()
        fig.savefig(figures / f'{path.parent.name}.png', dpi=180)
        plt.close(fig)
    if not runs:
        print('No completed training runs. LR range plots, if available, were exported.')
        return

    first = runs[0][2]
    for _, _, meta in runs:
        for key in ('source_sha256', 'gpu', 'torch', 'precision', 'split_seed', 'train_size'):
            if meta[key] != first[key]:
                raise ValueError(f'Mixed protocols/devices: {key}. Analyze separate output folders.')
        for key in ('protocol', 'depth', 'batch_size', 'bn_momentum', 'target_val_accuracy'):
            if meta['config'][key] != first['config'][key]:
                raise ValueError(f'Mixed configurations: {key}. Analyze separately.')
    groups = {}
    for _, summary, meta in runs:
        name = summary['experiment']
        if name in groups and groups[name][0][1]['config'] != meta['config']:
            raise ValueError(f'Different settings for {name}. Analyze separately.')
        groups.setdefault(name, []).append((summary, meta))
    output_rows = []
    for name, members in groups.items():
        scores = [100 * s['test_accuracy'] for s, _ in members]
        output_rows.append(dict(experiment=name, seeds=','.join(str(s['seed']) for s, _ in members),
            n=len(scores), test_accuracy_mean_pct=statistics.mean(scores),
            test_accuracy_sd_pp=statistics.stdev(scores) if len(scores) > 1 else '',
            elapsed_seconds_mean=statistics.mean(s['elapsed_seconds'] for s, _ in members),
            updates=members[0][0]['updates'],
            selected_epoch_mean=statistics.mean(s['selected_epoch'] for s, _ in members)))
    with (root / 'aggregate.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for path, summary, _ in runs:
        with (path / 'history.csv').open() as f:
            rows = list(csv.DictReader(f))
        label = f'{summary["experiment"]}, seed {summary["seed"]}'
        scores = [100 * float(r['val_accuracy']) for r in rows]
        axes[0].plot([int(r['step']) for r in rows], scores, label=label)
        axes[1].plot([float(r['elapsed_seconds']) / 60 for r in rows], scores, label=label)
    for ax in axes:
        ax.axhline(100 * first['config']['target_val_accuracy'], color='gray', linestyle='--', linewidth=1)
        ax.set(ylabel='Validation accuracy (%)')
        ax.grid(alpha=0.2)
    axes[0].set_xlabel('Optimizer updates')
    axes[1].set_xlabel('Training + validation time (minutes)')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / 'convergence.png', dpi=180)
    plt.close(fig)
    paired = []
    indexed = {(s['experiment'], s['seed']): s for _, s, _ in runs}
    for seed in sorted({s['seed'] for _, s, _ in runs}):
        baseline, cycle = indexed.get(('baseline_long', seed)), indexed.get(('onecycle', seed))
        if baseline is None or cycle is None:
            continue
        both_hit = baseline['seconds_to_target'] is not None and cycle['seconds_to_target'] is not None
        paired.append(dict(seed=seed,
            test_delta_pp=100*(cycle['test_accuracy']-baseline['test_accuracy']),
            budget_ratio=baseline['updates']/cycle['updates'],
            time_to_target_ratio=baseline['seconds_to_target']/cycle['seconds_to_target'] if both_hit else None,
            baseline_hit_target=baseline['seconds_to_target'] is not None,
            onecycle_hit_target=cycle['seconds_to_target'] is not None))
    (root / 'paired_comparison.json').write_text(json.dumps(paired, indent=2))
    lines = ['# Measured results', '',
             '| Method | Seeds | Test accuracy, % | SD, pp | Updates | Mean time, min |',
             '|---|---|---:|---:|---:|---:|']
    for r in output_rows:
        sd = f'{r["test_accuracy_sd_pp"]:.2f}' if r['n'] > 1 else 'unavailable (n=1)'
        lines.append(f'| {r["experiment"]} | {r["seeds"]} | {r["test_accuracy_mean_pct"]:.2f} | {sd} | {r["updates"]} | {r["elapsed_seconds_mean"]/60:.2f} |')
    lines.extend(['',
                  'Paired comparisons:', '```json', json.dumps(paired, indent=2), '```'])
    (root / 'results.md').write_text('\n'.join(lines))
    print(f'Exported {root / "aggregate.csv"}, results.md, paired_comparison.json and figures.')


if __name__ == '__main__':
    main()
