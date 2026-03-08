
import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_OUTPUT_SIZE = 32 * 10 * 2  # 640
HIDDEN_DIM = 64
BOTTLENECK_DIM = 8


class MyModel(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.mlp_bottleneck = nn.Sequential(
            nn.Linear(CONV_OUTPUT_SIZE, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, BOTTLENECK_DIM),
        )
        self.fc1 = nn.Linear(CONV_OUTPUT_SIZE + BOTTLENECK_DIM, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        x = F.relu(self.conv1(x))
        x_flat = x.view(bs, -1)
        x_bottleneck = self.mlp_bottleneck(x_flat)
        x_cat = torch.cat((x_flat, x_bottleneck), dim=-1)
        x = F.relu(self.fc1(x_cat))
        x = self.fc2(x)
        return x
