import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def datasets(root, synthetic=False):
    if synthetic:
        g = torch.Generator().manual_seed(2026)
        ds = TensorDataset(torch.randn(96, 3, 32, 32, generator=g),
                           torch.randint(0, 10, (96,), generator=g))
        return Subset(ds, range(64)), Subset(ds, range(64, 80)), Subset(ds, range(80, 96))
    from torchvision import datasets as tvd, transforms as T


    tvd.CIFAR10.url = 'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'

    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    eval_tf = T.Compose([T.ToTensor(), norm])
    augmented = tvd.CIFAR10(root, train=True, download=True, transform=train_tf)
    plain = tvd.CIFAR10(root, train=True, download=False, transform=eval_tf)
    labels = np.asarray(plain.targets)
    rng = np.random.default_rng(2026)
    train_idx, val_idx = [], []
    for label in range(10):
        idx = rng.permutation(np.flatnonzero(labels == label))
        val_idx.extend(idx[:500].tolist())
        train_idx.extend(idx[500:].tolist())
    test = tvd.CIFAR10(root, train=False, download=True, transform=eval_tf)
    return Subset(augmented, train_idx), Subset(plain, val_idx), test


def loader(ds, batch_size, workers, seed, shuffle, cuda):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=workers, pin_memory=cuda, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))
