"""Federated learning server for the supply-chain project."""
from __future__ import annotations

import os

import flwr as fl

from utils import (
    build_lstm_model,
    load_global_sequence_data,
    load_feature_cols_global,
    plotServerData,
)

FEATURE_COLS_FILE = "RES/feature_cols_global.txt"
TRANSFORM_META_FILE = "RES/transformation/transformer_stats.json"
GLOBAL_DATASET = "RES/transformation/global_transformed.csv"
WINDOW_SIZE = int(os.getenv("SEQ_WINDOW", "8"))
STEP_SIZE = int(os.getenv("SEQ_STEP", "1"))
GROUP_SPEC = os.getenv("SEQ_GROUP_COLS", "client,partition_market")
GROUP_COLUMNS = [c.strip() for c in GROUP_SPEC.split(",") if c.strip()]

feature_cols = load_feature_cols_global(FEATURE_COLS_FILE)
eval_X = None
eval_y = None
if os.path.exists(GLOBAL_DATASET):
    try:
        eval_X_tmp, eval_y_tmp, seq_feature_cols, seq_meta = load_global_sequence_data(
            GLOBAL_DATASET,
            FEATURE_COLS_FILE,
            transformer_meta_path=TRANSFORM_META_FILE,
            window_size=WINDOW_SIZE,
            step=STEP_SIZE,
            group_cols=GROUP_COLUMNS,
        )
        if eval_X_tmp.size and eval_y_tmp.size:
            eval_X = eval_X_tmp
            eval_y = eval_y_tmp
            feature_cols = seq_feature_cols
            print(
                f"[Server] Loaded global evaluation sequences: samples={len(eval_X)} window={WINDOW_SIZE} groups={len(seq_meta)}"
            )
        else:
            print("[Server] Global evaluation dataset lacks sufficient sequences; skipping evaluation.")
    except Exception as exc:  # pragma: no cover - optional evaluation
        print(f"[Server] Failed to build evaluation sequences: {exc}")

model = build_lstm_model(window_size=WINDOW_SIZE, n_features=len(feature_cols))
results_list: list[dict[str, float]] = []


def get_eval_fn(model, eval_data):
    Xs, ys = eval_data

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


strategy = fl.server.strategy.FedAvg(evaluate_fn=get_eval_fn(model, (eval_X, eval_y)))

fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=21),
    strategy=strategy,
)

plotServerData(results_list)
