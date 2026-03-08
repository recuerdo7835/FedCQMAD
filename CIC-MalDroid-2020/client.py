import os
import flwr as fl
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

#import utils.data_cnn as data_loader
import utils.data_qcnn as data_loader

#import utils.model_cnn as model_loader
#import utils.model_cmad as model_loader
import utils.model_qcnn_torchquantum as model_loader
#import utils.model_qcnn_torchquantum_gpu as model_loader


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Client(fl.client.NumPyClient):
    def __init__(self, client_id=None, num_clients=10, alpha=0.5, iid=False):

        self.X_train, self.Y_train, self.X_test, self.Y_test = data_loader.get_data(
            client_id=client_id,
            num_clients=num_clients,
            alpha=alpha,
            iid=iid,
        )
        n_classes = int(np.max(self.Y_train)) + 1
        self.model = model_loader.MyModel(num_classes=n_classes)
        #self.model = model_loader.OptimizedMyModel(num_classes=n_classes)  #torchquantum

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        n_classes = int(np.max(self.Y_train)) + 1
        class_counts = np.bincount(self.Y_train.astype(int), minlength=n_classes).astype(np.float32)
        class_weights = class_counts.sum() / (np.maximum(class_counts, 1.0) * n_classes)
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device))

        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        

        self.evaluation_count = 0
        self.total_rounds = 20

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, _):
        self.set_parameters(parameters)
        self.model.train()
        batch_size = 32
        num_epochs = 10
        running_loss = 0.0
        for epoch in range(num_epochs):
            num_batches = max(1, len(self.X_train) // batch_size)
            perm = np.random.permutation(len(self.X_train))
            for start in tqdm(range(0, len(self.X_train), batch_size), total=num_batches, desc=f"Training {epoch+1}/{num_epochs}"):
                idx = perm[start:start+batch_size]
                inputs = torch.tensor(self.X_train[idx], dtype=torch.float32).to(self.device)
                labels = torch.tensor(self.Y_train[idx], dtype=torch.long).to(self.device)
                if len(inputs) != batch_size:
                    continue
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
        avg_loss = running_loss / max(1, len(self.X_train) // batch_size)
        self.scheduler.step(avg_loss)
        return self.get_parameters({}), len(self.X_train), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        batch_size = 32
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = max(1, len(self.X_test) // batch_size)
        

        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.X_test), batch_size), total=num_batches, desc="Evaluating"):
                inputs = torch.tensor(self.X_test[i:i+batch_size], dtype=torch.float32).to(self.device)
                labels = torch.tensor(self.Y_test[i:i+batch_size], dtype=torch.long).to(self.device)
                if len(inputs) != batch_size:
                    continue
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        accuracy = correct / max(1, total)

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )
        

        cm = confusion_matrix(all_labels, all_predictions)
        

        roc_auc = 0.0
        if len(all_probs) > 0 and len(np.unique(all_labels)) >= 2:
            try:
                all_probs_arr = np.array(all_probs)
                roc_auc = roc_auc_score(all_labels, all_probs_arr, average="macro", multi_class="ovr")
            except Exception as e:
                print(f"ROC-AUC 计算失败: {e}")
        print(f"Correct: {correct}, Total: {total}, Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        print(f"Weighted Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
        

        self.evaluation_count += 1
        

        if self.evaluation_count >= self.total_rounds:
            print(f"第 {self.evaluation_count} 次评估，这是最终轮次，生成混淆矩阵")
            self._plot_confusion_matrix(cm, accuracy, precision, recall, f1)
        else:
            print(f"第 {self.evaluation_count} 次评估，不是最终轮次，跳过混淆矩阵生成")
        
        return total_loss / max(1, num_batches), len(self.X_test), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "roc_auc": roc_auc,
        }
    
    def _plot_confusion_matrix(self, cm, accuracy, precision, recall, f1):

        plt.figure(figsize=(10, 8))
        

        n_classes = len(cm)
        class_labels = [f"Class {i}" for i in range(n_classes)]
        

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()


        script_dir = os.path.abspath(os.path.dirname(__file__))
        results_dir = os.path.join(script_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "final_confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n>>> 最终混淆矩阵已保存: {save_path}")



if __name__ == "__main__":
    import time
    from flwr.client import start_client

    server_address = os.getenv("SERVER_ADDRESS", "localhost:8081")

    _client_id = os.getenv("CLIENT_ID")
    client_id = int(_client_id) if _client_id is not None else None
    num_clients = int(os.getenv("NUM_CLIENTS", "10"))
    alpha = float(os.getenv("DIRICHLET_ALPHA", "0.5"))
    iid = os.getenv("USE_IID", "0").strip().lower() in ("1", "true", "yes")
    client_instance = Client(client_id=client_id, num_clients=num_clients, alpha=alpha, iid=iid)
    client_instance_flower = client_instance.to_client()


    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            print(f"尝试连接服务器 {server_address}，第 {attempt+1} 次...")
            start_client(server_address=server_address, client=client_instance_flower)
            break  # 连接成功就跳出循环
        except Exception as e:
            print(f"连接失败: {e}")
            if attempt < max_attempts - 1:
                print("等待 3 秒后重试...")
                time.sleep(3)
            else:
                print("已达到最大重试次数，客户端启动失败。")

