
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):


    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 卷积后 32×6×2，展平后 32*6*2 = 384（无量子拼接）
        conv_output_size = 32 * 6 * 2

        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
