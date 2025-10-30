"""Utility helpers for the federated learning supply-chain demo."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

RESULTS_DIR = os.path.join("RES", "plots")


# =========================
# Feature helpers
# =========================

def load_feature_cols_global(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature columns file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cols = [line.strip() for line in f if line.strip()]
    return cols


def _align_to_global(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    aligned = pd.DataFrame(index=df.index)
    for col in feature_cols:
        aligned[col] = df[col] if col in df.columns else 0.0
    return aligned.astype(float)


def _load_transformation_meta(meta_path: str) -> Dict[str, Dict[str, float]]:
    if not meta_path or not os.path.exists(meta_path):
        return {"numeric_means": {}, "numeric_stds": {}}
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "numeric_means": {k: float(v) for k, v in data.get("numeric_means", {}).items()},
        "numeric_stds": {k: float(v) for k, v in data.get("numeric_stds", {}).items()},
    }


def _apply_meta_scaling(X: pd.DataFrame, meta: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    means = meta.get("numeric_means", {})
    stds = meta.get("numeric_stds", {})
    for col, mean in means.items():
        std = stds.get(col, 1.0) or 1.0
        if col in X.columns:
            X[col] = (X[col].astype(float) - mean) / std
    return X


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def get_client_data(
    client_features_path: str,
    feature_cols_file: str,
    test_size: float = 0.2,
    random_state: int = 42,
    transformer_meta_path: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feature_cols = load_feature_cols_global(feature_cols_file)
    df = pd.read_csv(client_features_path)
    drop_cols = [c for c in ["record_id", "client"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    y_col = "y_target_days_shipping"
    if y_col not in df.columns:
        raise ValueError(f"Target column '{y_col}' missing from {client_features_path}")
    y = df[y_col].astype(float).fillna(df[y_col].median())
    X = df.drop(columns=[y_col])
    X = _align_to_global(X, feature_cols)
    meta = _load_transformation_meta(transformer_meta_path or "")
    X = _apply_meta_scaling(X, meta)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y.values,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, y_train, X_test, y_test, feature_cols


def load_dataset_arrays(
    dataset_path: str, feature_cols_file: str, transformer_meta_path: str | None = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_cols = load_feature_cols_global(feature_cols_file)
    df = pd.read_csv(dataset_path)
    drop_cols = [c for c in ["record_id", "client"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    y_col = "y_target_days_shipping"
    if y_col not in df.columns:
        raise ValueError(f"Target column '{y_col}' missing from {dataset_path}")
    y = df[y_col].astype(float).fillna(df[y_col].median())
    X = df.drop(columns=[y_col])
    X = _align_to_global(X, feature_cols)
    meta = _load_transformation_meta(transformer_meta_path or "")
    X = _apply_meta_scaling(X, meta)
    return X.values, y.values, feature_cols


# =========================
# Sequence helpers
# =========================


def _infer_group_columns(df: pd.DataFrame, explicit: Sequence[str] | None = None) -> List[str]:
    if explicit:
        cols = [c for c in explicit if c in df.columns]
        if cols:
            return cols
    candidates = [
        "client",
        "partition_market",
        "partition_segment",
        "market",
        "Market",
    ]
    for cand in candidates:
        if cand in df.columns:
            return [cand]
    helper_col = "__group_key__"
    df[helper_col] = "Global"
    return [helper_col]


def _resolve_time_column(df: pd.DataFrame, priority: Sequence[str]) -> str:
    for col in priority:
        if col in df.columns:
            return col
    fallback = "__order_index__"
    df[fallback] = np.arange(len(df))
    return fallback


def _prepare_time_index(df: pd.DataFrame, time_col: str) -> pd.Series:
    base = pd.date_range("2000-01-01", periods=len(df), freq="H")
    if time_col == "__order_index__":
        return pd.Series(base, index=df.index)
    actual = pd.to_datetime(df[time_col], errors="coerce")
    time_index = pd.Series(base, index=df.index)
    mask = actual.notna()
    time_index.loc[mask] = actual.loc[mask]
    return time_index


def _window_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    window_size: int,
    step: int,
) -> Tuple[List[np.ndarray], List[float]]:
    seqs: List[np.ndarray] = []
    ys: List[float] = []
    if len(features) <= window_size:
        return seqs, ys
    end_limit = len(features) - window_size
    for start in range(0, end_limit, step):
        end = start + window_size
        target_idx = end
        if target_idx >= len(targets):
            break
        seqs.append(features[start:end])
        ys.append(float(targets[target_idx]))
    return seqs, ys


def _sequence_split(
    seqs: List[np.ndarray],
    targets: List[float],
    test_size: float,
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[float]]:
    if not seqs:
        return [], [], [], []
    if test_size <= 0:
        return seqs, targets, [], []
    n = len(seqs)
    test_count = max(1, int(np.ceil(n * test_size)))
    if test_count >= n:
        test_count = 1
    train_count = n - test_count
    if train_count <= 0:
        return [], [], seqs, targets
    train_seqs = seqs[:train_count]
    train_targets = targets[:train_count]
    test_seqs = seqs[train_count:]
    test_targets = targets[train_count:]
    return train_seqs, train_targets, test_seqs, test_targets


def _stack_or_empty(items: List[np.ndarray]) -> np.ndarray:
    if not items:
        return np.empty((0, 0, 0), dtype=np.float32)
    return np.stack([np.asarray(i, dtype=np.float32) for i in items]).astype(np.float32)


def _to_float_array(values: List[float]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


@dataclass
class SequenceDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_cols: List[str]
    metadata: Dict[str, Dict[str, int]]


def get_client_sequence_data(
    client_features_path: str,
    feature_cols_file: str,
    transformer_meta_path: str | None = None,
    window_size: int = 8,
    step: int = 1,
    test_size: float = 0.2,
    group_cols: Sequence[str] | None = None,
    time_column_priority: Sequence[str] = ("order_date", "week_start", "ship_date"),
    client_label: str | None = None,
) -> SequenceDataset:
    feature_cols = load_feature_cols_global(feature_cols_file)
    df = pd.read_csv(client_features_path)
    if client_label is not None and "client" not in df.columns:
        df["client"] = client_label
    df = df.copy()
    y_col = "y_target_days_shipping"
    if y_col not in df.columns:
        raise ValueError(f"Target column '{y_col}' missing from {client_features_path}")

    group_list = _infer_group_columns(df, group_cols)
    time_col = _resolve_time_column(df, time_column_priority)
    df["__time_key__"] = _prepare_time_index(df, time_col)

    meta = _load_transformation_meta(transformer_meta_path or "")
    train_seqs: List[np.ndarray] = []
    train_targets: List[float] = []
    test_seqs: List[np.ndarray] = []
    test_targets: List[float] = []
    per_group_counts: Dict[str, Dict[str, int]] = {}

    for group_values, sub in df.groupby(group_list, dropna=False):
        group_key = (
            group_values
            if isinstance(group_values, tuple)
            else (group_values,)
        )
        key_str = "::".join(str(v) if v == v else "Unknown" for v in group_key)
        sub = sub.sort_values("__time_key__").reset_index(drop=True)

        drop_cols = {"record_id", "client", "__time_key__", time_col}
        drop_cols.update(group_list)
        available_drop = [c for c in drop_cols if c in sub.columns]
        y = sub[y_col].astype(float).to_numpy()
        X = sub.drop(columns=[y_col])
        if available_drop:
            X = X.drop(columns=available_drop, errors="ignore")
        X = _align_to_global(X, feature_cols)
        X = _apply_meta_scaling(X, meta)

        seqs, targets = _window_sequences(X.to_numpy(dtype=float), y, window_size, step)
        per_group_counts[key_str] = {
            "rows": int(len(sub)),
            "sequences": int(len(seqs)),
        }
        if not seqs:
            continue
        grp_train, grp_train_y, grp_test, grp_test_y = _sequence_split(seqs, targets, test_size)
        if grp_train:
            train_seqs.extend(grp_train)
            train_targets.extend(grp_train_y)
        if grp_test:
            test_seqs.extend(grp_test)
            test_targets.extend(grp_test_y)

    X_train = _stack_or_empty(train_seqs)
    X_test = _stack_or_empty(test_seqs)
    y_train = _to_float_array(train_targets)
    y_test = _to_float_array(test_targets)

    return SequenceDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
        metadata=per_group_counts,
    )


def load_global_sequence_data(
    dataset_path: str,
    feature_cols_file: str,
    transformer_meta_path: str | None = None,
    window_size: int = 8,
    step: int = 1,
    group_cols: Sequence[str] | None = None,
    time_column_priority: Sequence[str] = ("order_date", "week_start", "ship_date"),
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Dict[str, int]]]:
    data = get_client_sequence_data(
        dataset_path,
        feature_cols_file,
        transformer_meta_path=transformer_meta_path,
        window_size=window_size,
        step=step,
        test_size=0.0,
        group_cols=group_cols,
        time_column_priority=time_column_priority,
    )
    X = np.concatenate([data.X_train, data.X_test], axis=0) if data.X_test.size else data.X_train
    y = np.concatenate([data.y_train, data.y_test], axis=0) if data.y_test.size else data.y_train
    return X, y, data.feature_cols, data.metadata


# =========================
# Model helpers
# =========================


def build_regression_model(n_features: int, learning_rate: float = 1e-3):
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - dependency error at runtime
        raise ImportError(
            "TensorFlow is required to build the regression model. Install tensorflow>=2.9."
        ) from exc

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="linear"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def build_lstm_model(
    window_size: int,
    n_features: int,
    learning_rate: float = 1e-3,
    lstm_units: int = 128,
    dense_units: int = 64,
):
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - dependency error at runtime
        raise ImportError(
            "TensorFlow is required to build the LSTM model. Install tensorflow>=2.9."
        ) from exc

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window_size, n_features)),
            keras.layers.Masking(mask_value=0.0),
            keras.layers.LSTM(lstm_units, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(dense_units, activation="relu"),
            keras.layers.Dense(1, activation="linear"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def get_training_callbacks():
    try:
        from tensorflow import keras
    except ImportError:  # pragma: no cover - same rationale as above
        return []
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=0
        ),
    ]


# =========================
# Plot helpers
# =========================

def _ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def plotServerData(results: Iterable[Dict[str, float]]) -> None:
    results = list(results)
    if not results:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - plotting optional
        return
    _ensure_results_dir()
    rounds = [r.get("round", idx) for idx, r in enumerate(results)]
    maes = [r.get("mae", float("nan")) for r in results]
    losses = [r.get("loss", float("nan")) for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds, maes, label="MAE", marker="o")
    ax.plot(rounds, losses, label="Loss", marker="s")
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Metric")
    ax.set_title("Server metrics")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(_ensure_results_dir(), "server_metrics.png")
    fig.savefig(out_path)
    plt.close(fig)


def plotClientData(results: Iterable[Dict[str, float]], client_name: str | None = None) -> None:
    results = list(results)
    if not results:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        return
    _ensure_results_dir()
    epochs = list(range(1, len(results) + 1))
    loss = [r.get("loss", float("nan")) for r in results]
    val_loss = [r.get("val_loss", float("nan")) for r in results]
    mae = [r.get("mae", float("nan")) for r in results]
    val_mae = [r.get("val_mae", float("nan")) for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, loss, label="Train Loss", marker="o")
    ax.plot(epochs, val_loss, label="Val Loss", marker="s")
    ax.set_xlabel("Local Epoch")
    ax.set_ylabel("Loss")
    title = f"Client metrics - {client_name}" if client_name else "Client metrics"
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper right")
    ax2 = ax.twinx()
    ax2.plot(epochs, mae, label="Train MAE", color="green", linestyle="--")
    ax2.plot(epochs, val_mae, label="Val MAE", color="red", linestyle="--")
    ax2.set_ylabel("MAE")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper center")
    fig.tight_layout()
    fname = "client_metrics"
    if client_name:
        safe_name = client_name.lower().replace(" ", "_")
        fname += f"_{safe_name}"
    out_path = os.path.join(_ensure_results_dir(), f"{fname}.png")
    fig.savefig(out_path)
    plt.close(fig)
