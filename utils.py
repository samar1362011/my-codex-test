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