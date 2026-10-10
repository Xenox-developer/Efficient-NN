import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.models import CifarResNet
from src.schedules import schedule


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    cfg = json.loads(Path('configs/kaggle.json').read_text())
    cycle = cfg['experiments']['onecycle']

    values = [schedule(cycle, i, 101) for i in range(101)]
    assert abs(values[0][0] - .1) < 1e-10
    assert abs(values[45][0] - 1.) < 1e-10
    assert abs(values[-1][0] - .0001) < 1e-10
    assert all(lr > 0 and .85 - 1e-12 <= momentum <= .95 + 1e-12 for lr, momentum in values)
    model = CifarResNet(56)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / name for name in ('continuous', 'resumed')]
        base = [sys.executable, '-m', 'src.train', '--experiment', 'onecycle', '--synthetic', '--device', 'cpu']
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        def run(extra):
            subprocess.run(base + extra, check=True, env=env)
        run(['--output', str(paths[0])])
        run(['--output', str(paths[1]), '--stop-after-epoch', '1'])
        assert not (paths[1] / 'onecycle_seed42' / 'summary.json').exists()
        run(['--output', str(paths[1]), '--resume'])
        ckpts = [torch.load(p / 'onecycle_seed42' / 'last.pt', weights_only=False) for p in paths]
        for key in ckpts[0]['model']:
            assert torch.equal(ckpts[0]['model'][key], ckpts[1]['model'][key]), key
        for a, b in zip(ckpts[0]['rows'], ckpts[1]['rows']):
            for key in ('step', 'lr', 'train_loss', 'val_loss', 'val_accuracy'):
                assert a[key] == b[key], key
        subprocess.run([sys.executable, '-m', 'src.train', '--experiment', 'lr_range', '--synthetic',
                        '--device', 'cpu', '--output', str(Path(tmp)/'range')], env=env, check=True)
    print('PASS: model output, LR/momentum schedule, training/evaluation, range test and exact CPU resume.')


if __name__ == '__main__':
    main()
