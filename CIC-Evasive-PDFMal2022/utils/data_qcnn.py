import os
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Non-IID 划分的默认参数（Dirichlet 分布）
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

    for df in (train_df, test_df):
        if "FileName" in df.columns:
            df.drop(columns=["FileName"], inplace=True)

    all_obj_cols = set(train_df.select_dtypes(include=["object"]).columns) | set(
        test_df.select_dtypes(include=["object"]).columns
    )
    obj_feature_cols = [c for c in all_obj_cols if c != label_col]

    for col in obj_feature_cols:
        le = LabelEncoder()
        merged = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
        le.fit(merged.fillna("<NA>"))
        if col in train_df.columns:
            train_df[col] = le.transform(train_df[col].astype(str).fillna("<NA>"))
        if col in test_df.columns:
            test_df[col] = le.transform(test_df[col].astype(str).fillna("<NA>"))

    y_le = LabelEncoder()
    merged_labels = pd.concat([train_df[label_col], test_df[label_col]], axis=0)
    y_le.fit(merged_labels.astype(str))
    y_train = y_le.transform(train_df[label_col].astype(str)).astype(np.int64)
    y_test = y_le.transform(test_df[label_col].astype(str)).astype(np.int64)

    X_train = train_df.drop(columns=[label_col])
    X_test = test_df.drop(columns=[label_col])
    common_cols = sorted(set(X_train.columns) | set(X_test.columns))
    X_train = X_train.reindex(columns=common_cols, fill_value=0.0)
    X_test = X_test.reindex(columns=common_cols, fill_value=0.0)

    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_test = scaler.transform(X_test.values)

    X_train = _reshape_for_qcnn(X_train, num_channels=3, width=2)
    X_test = _reshape_for_qcnn(X_test, num_channels=3, width=2)


    if client_id is not None:
        if iid:
            client_indices_list = iid_split_indices(len(y_train), num_clients, seed)
        else:
            client_indices_list = dirichlet_split_indices(y_train, num_clients, alpha, seed)
        idx = client_indices_list[client_id]
        X_train = X_train[idx]
        y_train = y_train[idx]

    return X_train, y_train, X_test, y_test


def _reshape_for_qcnn(X: np.ndarray, num_channels: int = 3, width: int = 2) -> np.ndarray:
    num_samples, num_features = X.shape
    block = num_channels * width
    required = int(np.ceil(num_features / block) * block)
    if required > num_features:
        X = np.pad(X, ((0, 0), (0, required - num_features)), mode="constant")
    height = required // block
    X = X.reshape(num_samples, num_channels, height, width)
    return X.astype(np.float32)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data()
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_test:", X_test.shape, "Y_test:", Y_test.shape)


