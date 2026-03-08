
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

CONV_OUTPUT_SIZE = 32 * 6 * 2  # 384
RANDOM_DIM = 128
BOTTLENECK_DIM = 8


class RandomFeatureLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale: float = 1.0):
        super().__init__()
        W = torch.randn(in_dim, out_dim) * scale
        self.register_buffer("W", W)
        b = torch.rand(out_dim) * 2 * np.pi
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(torch.matmul(x, self.W) + self.b)


class MyModel(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.random_features = RandomFeatureLayer(CONV_OUTPUT_SIZE, RANDOM_DIM, scale=0.1)
        self.fc_rf = nn.Linear(RANDOM_DIM, BOTTLENECK_DIM)
        self.fc1 = nn.Linear(CONV_OUTPUT_SIZE + BOTTLENECK_DIM, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        x = F.relu(self.conv1(x))
        x_flat = x.view(bs, -1)
        x_rf = self.random_features(x_flat)
        x_bottleneck = self.fc_rf(x_rf)
        x_cat = torch.cat((x_flat, x_bottleneck), dim=-1)
        x = F.relu(self.fc1(x_cat))
        x = self.fc2(x)
        return x
