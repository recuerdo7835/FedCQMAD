
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from math import ceil, pi

torch.manual_seed(0)

# 量子电路配置
n_qubits = 8
n_layers = 1


class QuantumCircuit(tq.QuantumModule):
    """优化的量子电路实现，减少计算开销"""

    def __init__(self):
        super().__init__()
        self.n_wires = n_qubits
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits))

    def forward(self, x, q_device: tq.QuantumDevice):
        batch_size = x.shape[0]
        for i in range(self.n_wires):
            tqf.ry(q_device, wires=i, params=x[:, i])
        for layer in range(n_layers):
            for i in range(self.n_wires):
                tqf.ry(q_device, wires=i, params=self.params[layer, i].expand(batch_size))
            for i in range(0, self.n_wires - 1, 2):
                tqf.cnot(q_device, wires=[i, i + 1])
        return q_device


class TorchQuantumLayer(tq.QuantumModule):
    """优化的量子层，支持批处理"""

    def __init__(self):
        super().__init__()
        self.n_wires = n_qubits
        self.circuit = QuantumCircuit()
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        batch_size = x.shape[0]
        self.q_device.reset_states(batch_size)
        self.circuit(x, self.q_device)
        result = self.measure(self.q_device)
        return result


class MyModel(nn.Module):
    """
    量子 CNN 模型（可学习低维投影版）
    唯一差异：在量子层前增加可学习线性投影 x_proj = x_cnn @ W_proj + b_proj ∈ R^8
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.qconv1 = TorchQuantumLayer()

        # 卷积展平后 32*79*2 = 5056
        conv_output_size = 32 * 79 * 2

        # 可学习低维投影：将高维特征映射到 8 维量子输入空间
        # x_proj = x_flat @ W_proj + b_proj ∈ R^8
        self.proj = nn.Linear(conv_output_size, n_qubits)

        self.fc1 = nn.Linear(conv_output_size + n_qubits, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        x = F.relu(self.conv1(x))
        x_flat = x.view(bs, -1)

        # 可学习低维投影（替代直接截取 x_flat[:, :8]）
        x_proj = self.proj(x_flat)  # (bs, 8)
        x_q = self.qconv1(x_proj)

        x_cat = torch.cat((x_flat, x_q), dim=-1)
        x = F.relu(self.fc1(x_cat))
        x = self.fc2(x)
        return x
