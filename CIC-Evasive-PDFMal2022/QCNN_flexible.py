
import os
import torch
import numpy as np
from tqdm import tqdm
import utils.data_qcnn as data_loader
import utils.model_qcnn_flexible as model_loader

CONV_OUTPUT_SIZE = 32 * 6 * 2  # PDF
DEFAULT_PROJ_DIM = 8
DEFAULT_N_QUBITS = 8


class Trainer:
    def __init__(self, num_classes: int, proj_dim: int, n_qubits: int):
        self.X_train, self.Y_train, self.X_test, self.Y_test = data_loader.get_data()
        self.model = model_loader.MyModel(
            num_classes=num_classes,
            conv_output_size=CONV_OUTPUT_SIZE,
            proj_dim=proj_dim,
            n_qubits=n_qubits,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model: proj_dim={proj_dim}, n_qubits={n_qubits}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        n_classes = int(self.Y_train.max() + 1)
        class_counts = np.bincount(self.Y_train, minlength=n_classes).astype(np.float32)
        class_weights = class_counts.sum() / (np.maximum(class_counts, 1.0) * n_classes)
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def train(self, num_epochs: int = 10, batch_size: int = 32):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss, correct, total = 0.0, 0, 0
            num_batches = max(1, len(self.X_train) // batch_size)
            perm = np.random.permutation(len(self.X_train))
            for start in tqdm(range(0, len(self.X_train), batch_size), total=num_batches, desc=f"Training {epoch+1}/{num_epochs}"):
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
        total_loss, correct, total = 0.0, 0, 0
        num_batches = max(1, len(self.X_test) // batch_size)
        with torch.no_grad():
            for i in tqdm(range(0, len(self.X_test), batch_size), total=num_batches, desc="Evaluating"):
                inputs = torch.tensor(self.X_test[i : i + batch_size], dtype=torch.float32).to(self.device)
                labels = torch.tensor(self.Y_test[i : i + batch_size], dtype=torch.long).to(self.device)
                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, labels).item()
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        print(f"Test Loss: {total_loss/max(1,num_batches):.4f} Test Acc: {correct/max(1,total):.4f}")


if __name__ == "__main__":
    proj_dim = int(os.getenv("PROJ_DIM", str(DEFAULT_PROJ_DIM)))
    n_qubits = int(os.getenv("N_QUBITS", str(DEFAULT_N_QUBITS)))
    print("QCNN (灵活投影+量子比特) - Evasive-PDFMal2022")
    print(f"proj_dim={proj_dim}, n_qubits={n_qubits}")
    trainer = Trainer(num_classes=2, proj_dim=proj_dim, n_qubits=n_qubits)
    trainer.train(num_epochs=10)
    trainer.evaluate()
