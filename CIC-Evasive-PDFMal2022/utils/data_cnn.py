import os
import re
import hashlib
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler



DEFAULT_NUM_CLIENTS = 10
DEFAULT_ALPHA = 0.5
DEFAULT_SEED = 42


def dirichlet_split_indices(y, num_clients, alpha, seed=None):

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
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


def iid_split_indices(n_samples, num_clients, seed=None):

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    perm = rng.permutation(n_samples)
    splits = np.array_split(perm, num_clients)
    return [np.array(s, dtype=np.int64) for s in splits]


def get_data(client_id=None, num_clients=DEFAULT_NUM_CLIENTS, alpha=DEFAULT_ALPHA, seed=DEFAULT_SEED, iid=False):

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(base_dir, "data", "train.csv")
    test_path = os.path.join(base_dir, "data", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_col = "Class"

    # 删除明显的标识列
    for df in (train_df, test_df):
        if "FileName" in df.columns:
            df.drop(columns=["FileName"], inplace=True)

    # 统一对象列集合（不含标签列）
    all_obj_cols = set(train_df.select_dtypes(include=["object"]).columns) | set(
        test_df.select_dtypes(include=["object"]).columns
    )
    obj_feature_cols = [c for c in all_obj_cols if c != label_col]

    # 使用确定性编码，避免跨客户端不一致
    train_df = _encode_object_columns_deterministically(train_df, obj_feature_cols)
    test_df = _encode_object_columns_deterministically(test_df, obj_feature_cols)

    # 标签编码（固定映射，确保一致）
    label_map = {"Benign": 0, "Malicious": 1}
    y_train = train_df[label_col].astype(str).map(label_map).astype(np.int64).to_numpy()
    y_test = test_df[label_col].astype(str).map(label_map).astype(np.int64).to_numpy()

    # 特征矩阵（统一列集合与顺序）
    X_train = train_df.drop(columns=[label_col])
    X_test = test_df.drop(columns=[label_col])
    common_cols = sorted(set(X_train.columns) | set(X_test.columns))
    X_train = X_train.reindex(columns=common_cols, fill_value=0.0)
    X_test = X_test.reindex(columns=common_cols, fill_value=0.0)

    # 转数值并填充
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_test = scaler.transform(X_test.values)

    # 统一重排为 3 通道、宽度 2
    X_train = _reshape_for_cnn(X_train, num_channels=3, width=2)
    X_test = _reshape_for_cnn(X_test, num_channels=3, width=2)

    # 客户端数据划分：IID（均匀随机）或 Non-IID（Dirichlet）
    if client_id is not None:
        if iid:
            client_indices_list = iid_split_indices(len(y_train), num_clients, seed)
        else:
            client_indices_list = dirichlet_split_indices(y_train, num_clients, alpha, seed)
        idx = client_indices_list[client_id]
        X_train = X_train[idx]
        y_train = y_train[idx]

    return X_train, y_train, X_test, y_test


def _reshape_for_cnn(X: np.ndarray, num_channels: int = 3, width: int = 2) -> np.ndarray:
    num_samples, num_features = X.shape
    block = num_channels * width
    required = int(np.ceil(num_features / block) * block)
    if required > num_features:
        X = np.pad(X, ((0, 0), (0, required - num_features)), mode="constant")
    height = required // block
    X = X.reshape(num_samples, num_channels, height, width)
    return X.astype(np.float32)


def _encode_object_columns_deterministically(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        series = df[col].astype(str).fillna("<NA>")
        # 常见二值映射
        series_lower = series.str.strip().str.lower()
        mask_yesno = series_lower.isin(["yes", "no"]) | series_lower.isin(["true", "false"]) | series_lower.isin(["1", "0"]) | series_lower.isin(["benign", "malicious"]) | series_lower.isin(["none"]) 
        mapped = series.copy()
        mapped[series_lower == "yes"] = 1
        mapped[series_lower == "true"] = 1
        mapped[series_lower == "1"] = 1
        mapped[series_lower == "no"] = 0
        mapped[series_lower == "false"] = 0
        mapped[series_lower == "0"] = 0
        mapped[series_lower == "benign"] = 0
        mapped[series_lower == "malicious"] = 1
        # Header 形如 "%PDF-1.4" → 1.4
        def parse_header(val: str) -> float:
            s = val.strip()
            m = re.search(r"%PDF-([0-9]+\.[0-9]+)", s)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return np.nan
            return np.nan
        if col.lower() == "header":
            num = series.map(parse_header)
            # 若解析失败，退化为哈希
            num = num.fillna(series.apply(_stable_hash_to_unit)).astype(float)
            df[col] = num
            continue
        # 其他值：已映射的保留，剩余用稳定哈希到 [0,1]
        remaining_mask = ~mask_yesno
        mapped.loc[remaining_mask] = series[remaining_mask].apply(_stable_hash_to_unit)
        df[col] = pd.to_numeric(mapped, errors="coerce").fillna(0.0)
    return df


def _stable_hash_to_unit(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    # 取前8位转为整数，映射到 [0,1]
    v = int(h[:8], 16)
    return (v % 1000000) / 1000000.0


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data()
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_test:", X_test.shape, "Y_test:", Y_test.shape)

