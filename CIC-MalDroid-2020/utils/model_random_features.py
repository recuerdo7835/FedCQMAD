
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

CONV_OUTPUT_SIZE = 32 * 79 * 2  # 5056
RANDOM_DIM = 128  # 随机特征维度
BOTTLENECK_DIM = 8


class RandomFeatureLayer(nn.Module):
    """随机傅里叶特征层：固定随机投影 + cos 非线性，近似 RBF 核"""

    def __init__(self, in_dim: int, out_dim: int, scale: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # 固定随机矩阵 W ~ N(0, scale^2)，不参与训练，形状 (in_dim, out_dim) 便于 x @ W
        W = torch.randn(in_dim, out_dim) * scale
        self.register_buffer("W", W)
        b = torch.rand(out_dim) * 2 * np.pi
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, in_dim), W: (in_dim, out_dim) -> z = cos(x @ W + b)
        # 使用 matmul 避免 F.linear 的 weight 形状歧义
        z = torch.cos(torch.matmul(x, self.W) + self.b)
        return z  # (bs, out_dim)


class MyModel(nn.Module):
    """
    经典 CNN + 随机特征映射
    Input -> Conv.ReLU -> Flatten 5056 -> [RandomFeatures(5056->128) -> Linear(128->8)] -> concat(5056,8)=5064 -> FC1.ReLU 128 -> FC2 5
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.random_features = RandomFeatureLayer(CONV_OUTPUT_SIZE, RANDOM_DIM, scale=0.1)
        self.fc_rf = nn.Linear(RANDOM_DIM, BOTTLENECK_DIM)  # 可学习线性层

        self.fc1 = nn.Linear(CONV_OUTPUT_SIZE + BOTTLENECK_DIM, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        x = F.relu(self.conv1(x))
        x_flat = x.view(bs, -1)
        x_rf = self.random_features(x_flat)  # (bs, 128)
        x_bottleneck = self.fc_rf(x_rf)  # (bs, 8)
        x_cat = torch.cat((x_flat, x_bottleneck), dim=-1)
        x = F.relu(self.fc1(x_cat))
        x = self.fc2(x)
        return x
