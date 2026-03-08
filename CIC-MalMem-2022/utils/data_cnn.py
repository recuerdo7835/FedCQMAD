import os
import glob
import re
import hashlib
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



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

    env_dir = os.getenv("MALMEM_DATA_DIR", "").strip()
    base_dir = env_dir if env_dir else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(base_dir, "data", "cic_malmem_train.csv")
    test_path = os.path.join(base_dir, "data", "cic_malmem_test.csv")
    mono_path = os.path.join(base_dir, "data", "Obfuscated-MalMem2022.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    elif os.path.exists(mono_path):
        df = pd.read_csv(mono_path)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        cut = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    else:

        data_dir = os.path.join(base_dir, "data")
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if len(csv_files) == 0:
            raise FileNotFoundError("未找到数据文件。请将 train.csv/test.csv 或 data.csv 放在 CIC-MalMem-2022/data/，或设置环境变量 MALMEM_DATA_DIR 指向包含 data 子目录的路径。")

        train_cands = [p for p in csv_files if "train" in os.path.basename(p).lower()]
        test_cands = [p for p in csv_files if "test" in os.path.basename(p).lower()]
        if len(train_cands) == 1 and len(test_cands) == 1:
            train_df = pd.read_csv(train_cands[0])
            test_df = pd.read_csv(test_cands[0])
        else:

            mono = max(csv_files, key=lambda p: os.path.getsize(p))
            df = pd.read_csv(mono)
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            cut = int(len(df) * 0.8)
            train_df, test_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()


    train_df.columns = [str(c).strip() for c in train_df.columns]
    test_df.columns = [str(c).strip() for c in test_df.columns]


    label_col = _detect_label_column(train_df, preferred=["Category", "Class", "Label", "Type"])  # 原始名，区分大小写

    if label_col not in test_df.columns:
        test_label_col = _detect_label_column(test_df, preferred=["Category", "Class", "Label", "Type"])  # 可能不同名
    else:
        test_label_col = label_col


    for df in (train_df, test_df):
        for col in ["FileName", "ID", "Id", "hash", "Hash"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)


    all_obj_cols = set(train_df.select_dtypes(include=["object"]).columns) | set(
        test_df.select_dtypes(include=["object"]).columns
    )
    obj_feature_cols = [c for c in all_obj_cols if c not in {label_col, test_label_col}]
    train_df = _encode_object_columns_deterministically(train_df, obj_feature_cols)
    test_df = _encode_object_columns_deterministically(test_df, obj_feature_cols)


    y_train_series, mask_train = _map_category_to_four_classes(train_df[label_col])
    y_test_series, mask_test = _map_category_to_four_classes(test_df[test_label_col])
    train_df = train_df[mask_train].reset_index(drop=True)
    test_df = test_df[mask_test].reset_index(drop=True)
    y_train = y_train_series.astype(np.int64).to_numpy()
    y_test = y_test_series.astype(np.int64).to_numpy()


    X_train = train_df.drop(columns=[label_col])
    X_test = test_df.drop(columns=[test_label_col])
    common_cols = sorted(set(X_train.columns) | set(X_test.columns))
    X_train = X_train.reindex(columns=common_cols, fill_value=0.0)
    X_test = X_test.reindex(columns=common_cols, fill_value=0.0)


    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_test = scaler.transform(X_test.values)


    X_train = _reshape_for_cnn(X_train, num_channels=3, width=2)
    X_test = _reshape_for_cnn(X_test, num_channels=3, width=2)


    if client_id is not None:
        if iid:
            client_indices_list = iid_split_indices(len(y_train), num_clients, seed)
        else:
            client_indices_list = dirichlet_split_indices(y_train, num_clients, alpha, seed)
        idx = client_indices_list[client_id]
        X_train = X_train[idx]
        y_train = y_train[idx]

    return X_train, y_train, X_test, y_test


def _map_category_to_four_classes(series: pd.Series) -> tuple[pd.Series, pd.Series]:

    s = series.astype(str).fillna("").str.strip().str.lower()
    label = pd.Series(index=s.index, dtype="float64")
    mapping = {
        "benign": 0,
        "ransomware": 1,
        "spyware": 2,
        "trojan": 3,
    }

    for key, idx in mapping.items():
        label[s.str.contains(key, regex=False)] = idx

    remaining = label.isna()
    if remaining.any():
        def prefix(t: str) -> str:
            m = re.match(r"^[a-z]+", t)
            if m:
                return m.group(0)
            return re.split(r"[\./_\-\s]", t)[0]
        pref = s[remaining].apply(prefix)
        for key, idx in mapping.items():
            mask = pref == key
            if mask.any():
                label.loc[mask.index[mask]] = idx
    valid_mask = label.notna()
    return label[valid_mask], valid_mask


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
        series_lower = series.str.strip().str.lower()

        mapped = series.copy()
        mapped[series_lower == "yes"] = 1
        mapped[series_lower == "true"] = 1
        mapped[series_lower == "1"] = 1
        mapped[series_lower == "no"] = 0
        mapped[series_lower == "false"] = 0
        mapped[series_lower == "0"] = 0

        for i in series.index:
            if series_lower.loc[i] not in {"yes", "true", "1", "no", "false", "0"}:
                mapped.loc[i] = _stable_hash_to_unit(series.loc[i])
        df[col] = pd.to_numeric(mapped, errors="coerce").fillna(0.0)
    return df


def _stable_hash_to_unit(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    v = int(h[:8], 16)
    return (v % 1000000) / 1000000.0


def _detect_label_column(df: pd.DataFrame, preferred: list[str]) -> str:

    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}
    for name in preferred:
        key = name.strip().lower()
        if key in lower_map:
            return lower_map[key]

    keywords = {"benign", "ransomware", "spyware", "trojan"}
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        values = df[c].astype(str).str.lower()
        hits = sum(values.str.contains(k, regex=False).any() for k in keywords)
        if hits >= 1:
            return c
    raise KeyError(f"未能自动识别标签列。可用列: {cols}. 请将标签列命名为 Category/Class/Label/Type 之一，或包含四类关键词。")


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data()
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_test:", X_test.shape, "Y_test:", Y_test.shape)


