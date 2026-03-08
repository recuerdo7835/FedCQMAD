
import torch
import torch.nn as nn
import torch.nn.functional as F

BOTTLENECK_DIM = 8
HIDDEN_DIM = 64


class MyModel(nn.Module):

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        conv_output_size = 32 * 79 * 2  # 5056

        # MLP 瓶颈：非线性低维映射，与 QCNN 的 proj+量子 对应
        self.mlp_bottleneck = nn.Sequential(
            nn.Linear(conv_output_size, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, BOTTLENECK_DIM),
        )

        self.fc1 = nn.Linear(conv_output_size + BOTTLENECK_DIM, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        x = F.relu(self.conv1(x))
        x_flat = x.view(bs, -1)
        x_bottleneck = self.mlp_bottleneck(x_flat)  # (bs, 8)
        x_cat = torch.cat((x_flat, x_bottleneck), dim=-1)
        x = F.relu(self.fc1(x_cat))
        x = self.fc2(x)
        return x
