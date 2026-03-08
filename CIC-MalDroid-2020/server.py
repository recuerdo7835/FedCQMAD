import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#import utils.data_cnn as data_loader
import utils.data_qcnn as data_loader

#import utils.model_cnn as model_loader
#import utils.model_cmad as model_loader
import utils.model_qcnn_torchquantum as model_loader
#import utils.model_qcnn_torchquantum_gpu as model_loader



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def weighted_average(metrics):
    total_examples = 0
    federated_metrics = {k: 0.0 for k in metrics[0][1].keys()}
    for num_examples, m in metrics:
        for k, v in m.items():
            federated_metrics[k] += num_examples * v
        total_examples += num_examples
    return {k: v / max(1, total_examples) for k, v in federated_metrics.items()}


def get_initial_parameters():

    _, y_train, _, _ = data_loader.get_data()
    n_classes = int(np.max(y_train)) + 1
    model = model_loader.MyModel(num_classes=n_classes)
    #model = model_loader.OptimizedMyModel(num_classes=n_classes)#torchquantum


    X_train, _, _, _ = data_loader.get_data()
    dummy = torch.tensor(X_train[:1], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)
    ndarrays = [val.detach().cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(ndarrays)


def _save_confusion_matrix_from_model(model, X_test, y_test, save_dir, batch_size=32):

    model.eval()
    device = next(model.parameters()).device
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.tensor(X_test[i : i + batch_size], dtype=torch.float32).to(device)
            out = model(batch_x)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_test[i : i + batch_size])
    cm = confusion_matrix(all_labels, all_preds)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    n_classes = len(cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Class {i}" for i in range(n_classes)],
                yticklabels=[f"Class {i}" for i in range(n_classes)])
    plt.title(f"Server Confusion Matrix (Global Model)\nAccuracy: {acc:.4f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "final_confusion_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def _make_evaluate_fn(num_rounds):

    _, _, X_test, y_test = data_loader.get_data()
    n_classes = int(np.max(y_test)) + 1
    server_dir = os.path.abspath(os.path.dirname(__file__))
    results_dir = os.path.join(server_dir, "results")

    def evaluate_fn(server_round, parameters, config):
        if server_round != num_rounds:
            return None
        try:
            ndarrays = parameters_to_ndarrays(parameters)
            model = model_loader.MyModel(num_classes=n_classes)
            state = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), ndarrays)}
            model.load_state_dict(state, strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            path = _save_confusion_matrix_from_model(model, X_test, y_test, results_dir)
            print(f"\n>>> 混淆矩阵已保存（服务器）: {path}")
        except Exception as e:
            print(f"服务器混淆矩阵保存失败: {e}")
        return None

    return evaluate_fn


class CustomFedAvgStrategy(fl.server.strategy.FedAvg):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_evaluate(self, server_round, parameters, client_manager):
        base_config = super().configure_evaluate(server_round, parameters, client_manager)
        if hasattr(self, "num_rounds") and server_round == self.num_rounds:
            print(f"第 {server_round} 轮是最终轮次，将生成混淆矩阵")
        return base_config


def get_server_strategy():
    num_rounds = 20
    strategy = CustomFedAvgStrategy(
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=get_initial_parameters(),
        evaluate_fn=_make_evaluate_fn(num_rounds),
    )
    strategy.num_rounds = num_rounds
    return strategy


if __name__ == "__main__":
    history = fl.server.start_server(
        server_address=os.getenv("SERVER_ADDRESS", "localhost:8081"),
        strategy=get_server_strategy(),
        config=fl.server.ServerConfig(num_rounds=20),
    )

    final_round = history.metrics_distributed["accuracy"][-1][0]
    print(f"\n=== 联邦学习完成，共训练 {final_round} 轮 ===")
    print("最终指标结果:")

    available_metrics = set()
    for metric_name in history.metrics_distributed.keys():
        if history.metrics_distributed[metric_name]:
            available_metrics.add(metric_name)

    for metric_name in sorted(available_metrics):
        if history.metrics_distributed[metric_name]:
            final_round, final_value = history.metrics_distributed[metric_name][-1]
            print(f"{metric_name.capitalize()}: {final_value:.4f}")

    if "accuracy" in history.metrics_distributed and history.metrics_distributed["accuracy"]:
        final_round, acc = history.metrics_distributed["accuracy"][-1]
        print(f"\n最终准确率: {acc:.3%}")

    server_dir = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(server_dir, "results", "final_confusion_matrix.png")
    print(f"\n混淆矩阵路径: {results_path}")


