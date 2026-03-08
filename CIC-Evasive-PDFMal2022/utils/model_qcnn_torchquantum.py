
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from math import ceil, pi

torch.manual_seed(0)

n_qubits = 8
n_layers = 1


class QuantumCircuit(tq.QuantumModule):
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
        return self.measure(self.q_device)


class MyModel(nn.Module):
    """量子 CNN（可学习低维投影版）- PDF 2 分类"""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.qconv1 = TorchQuantumLayer()
        conv_output_size = 32 * 6 * 2  # 3x6x2
        self.proj = nn.Linear(conv_output_size, n_qubits)
        self.fc1 = nn.Linear(conv_output_size + n_qubits, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        x = F.relu(self.conv1(x))
        x_flat = x.view(bs, -1)
        x_proj = self.proj(x_flat)
        x_q = self.qconv1(x_proj)
        x_cat = torch.cat((x_flat, x_q), dim=-1)
        x = F.relu(self.fc1(x_cat))
        x = self.fc2(x)
        return x
