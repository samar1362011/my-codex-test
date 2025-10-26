"""Federated learning server for the supply-chain project."""
from __future__ import annotations

import os

import flwr as fl

from utils import build_regression_model, load_dataset_arrays, load_feature_cols_global, plotServerData

FEATURE_COLS_FILE = "RES/feature_cols_global.txt"
TRANSFORM_META_FILE = "RES/transformation/transformer_stats.json"
GLOBAL_DATASET = "RES/transformation/global_transformed.csv"

feature_cols = load_feature_cols_global(FEATURE_COLS_FILE)
model = build_regression_model(n_features=len(feature_cols))
results_list: list[dict[str, float]] = []


def get_eval_fn(model):
    Xs, ys = None, None
    if os.path.exists(GLOBAL_DATASET):
        try:
            Xs, ys, _ = load_dataset_arrays(
                GLOBAL_DATASET, FEATURE_COLS_FILE, transformer_meta_path=TRANSFORM_META_FILE
            )
        except Exception as exc:  # pragma: no cover - evaluation optional
            print(f"[Server] Failed to load evaluation dataset: {exc}")
            Xs, ys = None, None

    def evaluate(server_round: int, parameters, config):
        model.set_weights(parameters)
        if Xs is None or ys is None or len(Xs) == 0:
            loss, mae = 0.0, float("nan")
        else:
            loss, mae = model.evaluate(Xs, ys, verbose=0)
        print(f"[Server] round={server_round} MAE={mae}")
        results_list.append({"round": server_round, "loss": float(loss), "mae": float(mae) if mae == mae else None})
        metrics = {"mae": 0.0 if mae != mae else float(mae)}
        return float(loss), metrics

    return evaluate


strategy = fl.server.strategy.FedAvg(evaluate_fn=get_eval_fn(model))

fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=21),
    strategy=strategy,
)

plotServerData(results_list)