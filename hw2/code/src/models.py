import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1, bn_momentum=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout, momentum=bn_momentum)
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False),
            nn.BatchNorm2d(cout, momentum=bn_momentum),
        )

    def forward(self, x):
        z = torch.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return torch.relu(z + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, depth=56, bn_momentum=0.1):
        super().__init__()
        if depth < 8 or (depth - 2) % 6:
            raise ValueError('depth must have the form 6n+2')
        n = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(16, momentum=bn_momentum), nn.ReLU())
        blocks, cin = [], 16
        for stage, cout in enumerate((16, 32, 64)):
            for i in range(n):
                blocks.append(Block(cin, cout, 2 if stage and i == 0 else 1, bn_momentum))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(64, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean((2, 3)))
