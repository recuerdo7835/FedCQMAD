"""
FedCMAD + 随机特征映射 本地训练 - MalDroid-2020
经典非线性特征映射：随机傅里叶特征（RBF 核近似）
模型内联，避免依赖 utils 同步问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import utils.data_qcnn as data_loader

torch.manual_seed(42)
np.random.seed(42)

CONV_OUTPUT_SIZE = 32 * 79 * 2
RANDOM_DIM = 128
BOTTLENECK_DIM = 8


class RandomFeatureLayer(nn.Module):
    """随机傅里叶特征层：固定随机投影 + cos 非线性"""

    def __init__(self, in_dim: int, out_dim: int, scale: float = 1.0):
        super().__init__()
        W = torch.randn(in_dim, out_dim) * scale
        self.register_buffer("W", W)
        b = torch.rand(out_dim) * 2 * np.pi
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cos(torch.matmul(x, self.W) + self.b)
        return z


class MyModel(nn.Module):
    def __init__(self, num_classes: int = 5):
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


class Trainer:
    def __init__(self, num_classes: int):
        self.X_train, self.Y_train, self.X_test, self.Y_test = data_loader.get_data()
        self.model = MyModel(num_classes=num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"Model moved to device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name()}")

        n_classes = int(self.Y_train.max() + 1)
        class_counts = np.bincount(self.Y_train, minlength=n_classes).astype(np.float32)
        class_weights = class_counts.sum() / (np.maximum(class_counts, 1.0) * n_classes)
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def train(self, num_epochs: int = 20, batch_size: int = 32):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            num_batches = max(1, len(self.X_train) // batch_size)
            perm = np.random.permutation(len(self.X_train))
            for start in tqdm(
                range(0, len(self.X_train), batch_size),
                total=num_batches,
                desc=f"Training {epoch+1}/{num_epochs}",
            ):
                idx = perm[start : start + batch_size]
                inputs = torch.tensor(self.X_train[idx], dtype=torch.float32).to(self.device)
                labels = torch.tensor(self.Y_train[idx], dtype=torch.long).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
            print(f"Epoch {epoch+1}/{num_epochs} Loss: {running_loss/num_batches:.4f} Acc: {correct/max(1,total):.4f}")

    def evaluate(self, batch_size: int = 32):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = max(1, len(self.X_test) // batch_size)
        with torch.no_grad():
            for i in tqdm(range(0, len(self.X_test), batch_size), total=num_batches, desc="Evaluating"):
                inputs = torch.tensor(
                    self.X_test[i : i + batch_size], dtype=torch.float32
                ).to(self.device)
                labels = torch.tensor(
                    self.Y_test[i : i + batch_size], dtype=torch.long
                ).to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        print(f"Test Loss: {total_loss/max(1,num_batches):.4f} Test Acc: {correct/max(1,total):.4f}")


if __name__ == "__main__":
    print("CMAD + Random Features Training Started...")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    trainer = Trainer(num_classes=5)
    trainer.train(num_epochs=20)
    trainer.evaluate()
