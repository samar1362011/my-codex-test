"""Flower client 2 for the federated learning experiment."""
from __future__ import annotations

import os

import flwr as fl

from utils import (
    build_regression_model,
    get_client_data,
    get_training_callbacks,
    plotClientData,
)

FEATURE_COLS_FILE = "RES/feature_cols_global.txt"
TRANSFORM_META_FILE = "RES/transformation/transformer_stats.json"
DEFAULT_CLIENT_NAME = "US"

client_name = os.getenv("CLIENT_NAME", DEFAULT_CLIENT_NAME)
client_path_override = os.getenv("CLIENT_DATA_PATH")

if client_path_override:
    client_features_path = client_path_override
else:
    transformed_default = os.path.join("RES", "transformation", client_name, "transformed_features.csv")
    raw_default = os.path.join("RES", "clients", client_name, "features.csv")
    client_features_path = transformed_default if os.path.exists(transformed_default) else raw_default

X_train, y_train, X_test, y_test, feature_cols = get_client_data(
    client_features_path,
    FEATURE_COLS_FILE,
    test_size=0.25,
    random_state=42,
    transformer_meta_path=TRANSFORM_META_FILE,
)

model = build_regression_model(n_features=len(feature_cols))
callbacks = get_training_callbacks()
results_list: list[dict[str, float]] = []


class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_name: str = "Client 2"):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_name = client_name
        self.callbacks = callbacks

    def get_properties(self, config):
        return {"client_name": self.client_name, "n_features": int(self.X_train.shape[1])}

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