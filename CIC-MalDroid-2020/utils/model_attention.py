"""
经典轻量注意力层 - MalDroid-2020
在 model_cmad 基础上增加通道注意力：对卷积特征做全局池化 + 注意力式非线性映射得到 8 维摘要
类似 Squeeze-and-Excitation 的简化版，输出 8 维与 QCNN 量子输出维度一致
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_OUTPUT_SIZE = 32 * 79 * 2  # 5056
NUM_CHANNELS = 32
BOTTLENECK_DIM = 8


class ChannelAttentionBottleneck(nn.Module):
    """通道注意力瓶颈：GlobalPool(32ch) -> 32 -> Linear(32,8) -> ReLU -> 8 维摘要"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(NUM_CHANNELS, BOTTLENECK_DIM * 2),
            nn.ReLU(),
            nn.Linear(BOTTLENECK_DIM * 2, BOTTLENECK_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, 32, 79, 2)
        bs = x.size(0)
        x_pool = x.view(bs, NUM_CHANNELS, -1).mean(dim=2)  # (bs, 32) 全局平均池化
        return self.fc(x_pool)  # (bs, 8)


class MyModel(nn.Module):
    """
    经典 CNN + 通道注意力瓶颈
    Input 3×79×2 -> Conv.ReLU 32×79×2 -> [ChannelAttn: pool->32->16->ReLU->8] -> concat(5056,8)=5064 -> FC1.ReLU 128 -> FC2 5
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.attn_bottleneck = ChannelAttentionBottleneck()

        self.fc1 = nn.Linear(CONV_OUTPUT_SIZE + BOTTLENECK_DIM, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        x = F.relu(self.conv1(x))
        x_flat = x.view(bs, -1)
        x_attn = self.attn_bottleneck(x)  # (bs, 8)，在 flatten 前用通道维度
        x_cat = torch.cat((x_flat, x_attn), dim=-1)
        x = F.relu(self.fc1(x_cat))
        x = self.fc2(x)
        return x
