"""Flower client 1 for the federated learning experiment."""
from __future__ import annotations

import os
from pathlib import Path

import flwr as fl

from utils import (
    build_lstm_model,
    get_client_sequence_data,
    get_training_callbacks,
    plotClientData,
)

FEATURE_COLS_FILE = "RES/feature_cols_global.txt"
TRANSFORM_META_FILE = "RES/transformation/transformer_stats.json"
DEFAULT_CLIENT_NAME = "Europe"
WINDOW_SIZE = int(os.getenv("SEQ_WINDOW", "8"))
STEP_SIZE = int(os.getenv("SEQ_STEP", "1"))
GROUP_SPEC = os.getenv("SEQ_GROUP_COLS", "client,partition_market")
GROUP_COLUMNS = [c.strip() for c in GROUP_SPEC.split(",") if c.strip()]

client_name = os.getenv("CLIENT_NAME", DEFAULT_CLIENT_NAME)
client_path_override = os.getenv("CLIENT_DATA_PATH")

if client_path_override:
    client_features_path = client_path_override
else:
    transformed_default = os.path.join("RES", "transformation", client_name, "transformed_features.csv")
    raw_default = os.path.join("RES", "clients", client_name, "features.csv")
    client_features_path = transformed_default if os.path.exists(transformed_default) else raw_default

candidate_message: list[str] = []

if not Path(client_features_path).exists():
    candidates: list[Path] = []
    clients_dir = Path("RES") / "clients"
    if clients_dir.exists():
        candidates.extend(sorted(clients_dir.glob("*/features.csv")))

    transformed_dir = Path("RES") / "transformation"
    if transformed_dir.exists():
        candidates.extend(sorted(transformed_dir.glob("*/transformed_features.csv")))

    candidate_message.append(
        "Client feature file not found: "
        f"{client_features_path}."
    )

    if candidates:
        candidate_message.append("Available feature files:")
        for cand in candidates:
            candidate_message.append(f"  - {cand.as_posix()}")
        candidate_message.append(
            "Set CLIENT_NAME to one of the directories above or point CLIENT_DATA_PATH to a specific file."
        )
    else:
        candidate_message.append(
            "No client features were found under RES/. Run the preprocessing pipeline (python pipeline.py) "
            "to generate them, or provide CLIENT_DATA_PATH pointing to an existing CSV."
        )

    raise SystemExit("\n".join(candidate_message))

sequence_data = get_client_sequence_data(
    client_features_path,
    FEATURE_COLS_FILE,
    transformer_meta_path=TRANSFORM_META_FILE,
    window_size=WINDOW_SIZE,
    step=STEP_SIZE,
    test_size=0.25,
    group_cols=GROUP_COLUMNS,
    client_label=client_name,
)

if sequence_data.X_train.size == 0 or sequence_data.X_test.size == 0:
    raise SystemExit(
        "Not enough sequential data to train an LSTM. "
        "Reduce SEQ_WINDOW/SEQ_STEP or ensure the client dataset has at least "
        f"{WINDOW_SIZE + 1} chronological rows per group."
    )

X_train, y_train = sequence_data.X_train, sequence_data.y_train
X_test, y_test = sequence_data.X_test, sequence_data.y_test
feature_cols = sequence_data.feature_cols

model = build_lstm_model(window_size=WINDOW_SIZE, n_features=len(feature_cols))
callbacks = get_training_callbacks()
results_list: list[dict[str, float]] = []


class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_name: str = "Client 1"):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_name = client_name
        self.callbacks = callbacks

    def get_properties(self, config):
        return {
            "client_name": self.client_name,
            "n_features": int(self.X_train.shape[-1]),
            "window_size": WINDOW_SIZE,
        }

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=64,
            epochs=3,
            validation_data=(self.X_test, self.y_test),
            callbacks=self.callbacks,
            verbose=0,
        )
        ep = -1
        results = {
            "loss": float(history.history["loss"][ep]),
            "mae": float(history.history["mae"][ep]),
            "val_loss": float(history.history["val_loss"][ep]),
            "val_mae": float(history.history["val_mae"][ep]),
        }
        print(f"[{self.client_name}] local metrics: {results}")
        results_list.append(results)
        return self.model.get_weights(), len(self.X_train), results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"[{self.client_name}] eval MAE after aggregation: {mae}")
        return float(loss), len(self.X_test), {"mae": float(mae)}


client = FlwrClient(model, X_train, y_train, X_test, y_test, client_name=f"Client - {client_name}")
fl.client.start_numpy_client(server_address="localhost:8080", client=client)
plotClientData(results_list, client_name=client_name)
