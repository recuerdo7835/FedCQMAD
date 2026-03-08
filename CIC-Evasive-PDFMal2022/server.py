import flwr as fl
from flwr.common import ndarrays_to_parameters
import os
import numpy as np
import torch

#import utils.data_cnn as data_loader
import utils.data_qcnn as data_loader

#import utils.model_cnn as model_loader
import utils.model_cmad as model_loader
#import utils.model_qcnn_torchquantum as model_loader
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
    # 基于数据自动确定类别数，构建与客户端一致的 QCNN，并导出初始参数
    _, y_train, _, _ = data_loader.get_data()
    n_classes = int(np.max(y_train)) + 1
    model = model_loader.MyModel(num_classes=n_classes)
    #model = model_loader.OptimizedMyModel(num_classes=n_classes)#torchquantum

    # 通过一次前向传播初始化 Lazy 模块参数
    X_train, _, _, _ = data_loader.get_data()
    dummy = torch.tensor(X_train[:1], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)
    ndarrays = [val.detach().cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(ndarrays)


def get_server_strategy():
    return fl.server.strategy.FedAvg(
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=get_initial_parameters(),
    )


if __name__ == "__main__":
    history = fl.server.start_server(
        server_address=os.getenv("SERVER_ADDRESS", "localhost:8080"),
        strategy=get_server_strategy(),
        config=fl.server.ServerConfig(num_rounds=20),
    )
    
    # 显示最终轮次的所有指标
    final_round = history.metrics_distributed["accuracy"][-1][0]
    
    print(f"\n=== 联邦学习完成，共训练 {final_round} 轮 ===")
    print(f"最终指标结果:")
    
    # 获取所有可用的指标
    available_metrics = set()
    for metric_name in history.metrics_distributed.keys():
        if history.metrics_distributed[metric_name]:
            available_metrics.add(metric_name)
    
    # 显示每个指标的最终值
    for metric_name in sorted(available_metrics):
        if history.metrics_distributed[metric_name]:
            final_round, final_value = history.metrics_distributed[metric_name][-1]
            print(f"{metric_name.capitalize()}: {final_value:.4f}")
    
    # 特别显示准确率
    if "accuracy" in history.metrics_distributed and history.metrics_distributed["accuracy"]:
        final_round, acc = history.metrics_distributed["accuracy"][-1]
        print(f"\n最终准确率: {acc:.3%}")
    
    print(f"\n所有指标历史记录已保存在 history 对象中")
    print("最终混淆矩阵图已保存到 results/final_confusion_matrix_fedcnn_5.png")


