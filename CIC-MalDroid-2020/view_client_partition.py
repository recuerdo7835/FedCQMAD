
import os
import sys
from collections import defaultdict

# 确保能导入 utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import utils.data_qcnn as data_loader

# 可调参数
NUM_CLIENTS = 10
ALPHA = 0.5
IID = False# True=查看 IID 划分, False=查看 Non-IID (Dirichlet) 划分
SEED = 42


def _dirichlet_split_indices(y, num_clients, alpha, seed):
    """Dirichlet 划分，本脚本自包含实现"""
    rng = np.random.default_rng(seed)
    n_classes = int(np.max(y)) + 1
    idx_by_class = defaultdict(list)
    for i, yi in enumerate(y):
        idx_by_class[int(yi)].append(i)
    client_indices = [[] for _ in range(num_clients)]
    for k in range(n_classes):
        indices = np.array(idx_by_class[k])
        if len(indices) == 0:
            continue
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.ones(num_clients) * alpha)
        proportions = proportions / proportions.sum()
        counts = rng.multinomial(len(indices), proportions)
        start = 0
        for i in range(num_clients):
            end = start + counts[i]
            if end > start:
                client_indices[i].extend(indices[start:end])
            start = end
    return [np.array(ci, dtype=np.int64) for ci in client_indices]


def _iid_split_indices(n_samples, num_clients, seed):

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    splits = np.array_split(perm, num_clients)
    return [np.array(s, dtype=np.int64) for s in splits]


def main():
    # 加载全量数据（仅需 get_data() 无参调用）
    _, y_full, _, _ = data_loader.get_data()
    n_classes = int(np.max(y_full)) + 1

    # 本地划分索引（自包含实现，不依赖 data_qcnn 的 client_id 参数）
    if IID:
        client_indices_list = _iid_split_indices(len(y_full), NUM_CLIENTS, SEED)
    else:
        client_indices_list = _dirichlet_split_indices(
            y_full, NUM_CLIENTS, ALPHA, SEED
        )

    mode = "IID (均匀随机)" if IID else "Non-IID (Dirichlet alpha={})".format(ALPHA)
    print("=" * 70)
    print("{} 数据划分详情 ({} 客户端)".format(mode, NUM_CLIENTS))
    print("=" * 70)

    # 表头
    header = "Client | 总样本数 | " + " | ".join("Class {}".format(k) for k in range(n_classes)) + " | 占比"
    print(header)
    print("-" * 70)

    total_samples = len(y_full)
    all_counts = []

    for cid in range(NUM_CLIENTS):
        idx = client_indices_list[cid]
        y_c = y_full[idx]
        counts = np.bincount(y_c, minlength=n_classes)
        all_counts.append(counts)
        n = len(y_c)
        pct = 100.0 * n / total_samples if total_samples > 0 else 0
        row = "  {}    | {:>6}   | ".format(cid, n) + " | ".join("{:>6}".format(c) for c in counts) + " | {:.1f}%".format(pct)
        print(row)

    print("-" * 70)
    print("全量训练集: {} 样本, {} 类".format(total_samples, n_classes))
    print()


    print("各类别在各客户端的分布（每行=一个类别）:")
    print("-" * 70)
    for k in range(n_classes):
        class_total = sum(all_counts[i][k] for i in range(NUM_CLIENTS))
        row = "Class {} (共{:>4}): ".format(k, class_total) + " | ".join(
            "C{}:{:>4}".format(i, all_counts[i][k]) for i in range(NUM_CLIENTS)
        )
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    main()
