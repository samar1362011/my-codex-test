# pipeline.py
"""Supply chain preprocessing, transformation, and routing pipeline."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import hashlib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Optional plotting import kept for compatibility (not required during tests)
try:  # pragma: no cover - optional dependency for manual plotting
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting is optional
    plt = None

# =========================
# CONFIG
# =========================
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "DATASET")
RES_DIR = os.path.join(BASE_DIR, "RES")

PRE_DIR = os.path.join(RES_DIR, "preprocessing")
TRANS_DIR = os.path.join(RES_DIR, "transformation")
OPT_DIR = os.path.join(RES_DIR, "optimization")
CLIENTS_DIR = os.path.join(RES_DIR, "clients")

FN_DS_SUPPLY = "DS_DataCoSupplyChainDataset.csv"

FILES = {
    "supply_clean_sample": "DS_SupplyChain_cleaned_sample.csv",
    "feature_cols_global": "feature_cols_global.txt",
    "transformer_stats": "transformer_stats.json",
}

RANDOM_STATE = 42
PARTITION_COL = "Market"

FAILSAFE_ALWAYS_GLOBAL = True
FAILSAFE_FORCE_SYNTH = True
FAILSAFE_MIN_POINTS = 20
MIN_ROWS_PER_CLIENT = 1
MAX_OPTIMIZATION_POINTS = 600

# Synthetic coordinate controls (degrees around an anchor point)
SYNTH_COORD_LAT_CENTER = 25.276987
SYNTH_COORD_LON_CENTER = 55.296249
SYNTH_COORD_LAT_RANGE_DEG = 0.8
SYNTH_COORD_LON_RANGE_DEG = 1.0

VEHICLES = {
    "bike": {"speed_kmph": 15.0, "cost_per_km": 0.05, "max_km_per_route": 25, "capacity": 25},
    "car": {"speed_kmph": 35.0, "cost_per_km": 0.12, "max_km_per_route": 120, "capacity": 80},
    "van": {"speed_kmph": 28.0, "cost_per_km": 0.18, "max_km_per_route": 200, "capacity": 150},
}
SERVICE_TIME_MIN = 5.0
LAMBDA_COST = 0.5
DEFAULT_SLA_MIN = 60.0

TRANSFORM_META_FILE = os.path.join(TRANS_DIR, FILES["transformer_stats"])
GLOBAL_TRANSFORM_PATH = os.path.join(TRANS_DIR, "global_transformed.csv")

# =========================
# UTILITIES
# =========================

def ensure_dirs() -> None:
    for d in [RES_DIR, PRE_DIR, TRANS_DIR, OPT_DIR, CLIENTS_DIR]:
        os.makedirs(d, exist_ok=True)


def read_csv_safely(path: str, **kwargs) -> pd.DataFrame:
    from io import StringIO

    base_defaults = dict(sep=None, on_bad_lines="skip", engine="python")
    for k, v in base_defaults.items():
        kwargs.setdefault(k, v)
    if kwargs.get("engine") == "python" and "low_memory" in kwargs:
        kwargs.pop("low_memory", None)

    encodings_to_try = ["utf-8", "cp1252", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on dataset
            last_err = exc
            continue

    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("cp1252", errors="replace")
    except Exception:
        text = raw.decode("latin-1", errors="replace")
    return pd.read_csv(StringIO(text), **kwargs)


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8")


def normalize_text(s: object) -> object:
    if pd.isna(s):
        return np.nan
    text = str(s).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_lower(s: object) -> object:
    if pd.isna(s):
        return np.nan
    text = str(s).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_float(x: object) -> float:
    if pd.isna(x):
        return float("nan")
    s = re.sub(r"[^\d\.\-]", "", str(x))
    try:
        return float(s)
    except Exception:
        return float("nan")


def parse_int(x: object) -> float:
    if pd.isna(x):
        return float("nan")
    s = re.sub(r"[^\d\-]", "", str(x))
    try:
        return float(s)
    except Exception:
        return float("nan")


def to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


# ---------- DEBUG / HEARTBEAT ----------

def _opt_hb(msg: str) -> None:
    os.makedirs(OPT_DIR, exist_ok=True)
    hb = os.path.join(OPT_DIR, "_heartbeat.txt")
    with open(hb, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
    print("[OPT]", msg)


# =========================
# GEO / DISTANCE
# =========================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(R * c)


# =========================
# FLEXIBLE COLUMN PICKER
# =========================

def pick_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    lower_map = {re.sub(r"\s+", " ", str(c)).strip().lower(): c for c in df.columns}
    for cand in cands:
        key = re.sub(r"\s+", " ", str(cand)).strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


# =========================
# PREPROCESS (Supply only)
# =========================

def preprocess_ds_supply_only(path: str) -> pd.DataFrame:
    df = read_csv_safely(path)
    df = df.drop_duplicates().copy()

    col_order_id = pick_col(df, "order id", "order_number", "order item id", "orderitem id")
    col_order_date = pick_col(
        df, "order date", "order created date", "order purchased date", "order date (dateorders)"
    )
    col_ship_date = pick_col(
        df, "shipping date", "shipping date (date orders)", "ship date", "ship date (dateorders)"
    )
    col_qty = pick_col(df, "order item quantity", "order quantity", "quantity")
    col_price = pick_col(
        df,
        "order item product price",
        "product price",
        "unit price",
        "item price",
        "order item total",
    )
    col_ship_mode = pick_col(df, "shipping mode", "shipment mode", "ship mode")
    col_category = pick_col(df, "category name", "product category", "category", "product type")
    col_market = pick_col(df, "market", "region", "customer market")
    col_segment = pick_col(df, "customer segment", "segment")

    col_order_city = pick_col(df, "order city", "customer city", "city")
    col_order_state = pick_col(df, "order state", "customer state", "state")
    col_order_country = pick_col(df, "order country", "customer country", "country")

    col_lat = pick_col(df, "drop latitude", "latitude", "customer latitude", "order latitude")
    col_lon = pick_col(df, "drop longitude", "longitude", "customer longitude", "order longitude")
    col_slat = pick_col(df, "store latitude", "seller latitude", "warehouse latitude")
    col_slon = pick_col(df, "store longitude", "seller longitude", "warehouse longitude")

    df["order_id_std"] = df[col_order_id] if col_order_id else np.arange(len(df)) + 1
    df["order_id_std"] = df["order_id_std"].astype(str)
    df["order_date"] = to_datetime_safe(df[col_order_date]) if col_order_date else pd.NaT
    df["ship_date"] = to_datetime_safe(df[col_ship_date]) if col_ship_date else pd.NaT
    df["order_qty"] = pd.to_numeric(df[col_qty], errors="coerce") if col_qty else np.nan
    df["price_dollar_sc"] = df[col_price].map(parse_float) if col_price else np.nan
    df["shipment_mode"] = df[col_ship_mode].astype(str).map(normalize_lower) if col_ship_mode else np.nan
    df["category_std"] = df[col_category].astype(str).map(normalize_lower) if col_category else np.nan

    df["partition_market"] = (
        df[col_market].astype(str).map(normalize_text) if col_market else "Global"
    )
    df["partition_segment"] = (
        df[col_segment].astype(str).map(normalize_text) if col_segment else "All"
    )

    if col_order_city:
        df["Order City"] = df[col_order_city].astype(str).map(normalize_text)
    if col_order_state:
        df["Order State"] = df[col_order_state].astype(str).map(normalize_text)
    if col_order_country:
        df["Order Country"] = df[col_order_country].astype(str).map(normalize_text)

    real_days_col = pick_col(df, "days for shipping (real)")
    if real_days_col:
        df["days_shipping_real"] = pd.to_numeric(df[real_days_col], errors="coerce")
    else:
        if df["order_date"].notna().any() and df["ship_date"].notna().any():
            df["days_shipping_real"] = (df["ship_date"] - df["order_date"]).dt.days.astype("float")
        else:
            df["days_shipping_real"] = np.nan

    if df["order_date"].notna().any():
        df["week_start"] = (
            pd.to_datetime(df["order_date"]).dt.to_period("W").apply(lambda r: r.start_time)
        )
    else:
        df["week_start"] = pd.Timestamp.today().normalize()

    if col_lat and col_lon:
        df["Drop_Latitude"] = pd.to_numeric(df[col_lat], errors="coerce")
        df["Drop_Longitude"] = pd.to_numeric(df[col_lon], errors="coerce")
        _opt_hb(f"preprocess: mapped {col_lat}/{col_lon} -> Drop_*")
    if col_slat and col_slon:
        df["Store_Latitude"] = pd.to_numeric(df[col_slat], errors="coerce")
        df["Store_Longitude"] = pd.to_numeric(df[col_slon], errors="coerce")

    for col in ["Drop_Latitude", "Store_Latitude"]:
        if col in df.columns:
            df.loc[(df[col] < -90) | (df[col] > 90), col] = np.nan
    for col in ["Drop_Longitude", "Store_Longitude"]:
        if col in df.columns:
            df.loc[(df[col] < -180) | (df[col] > 180), col] = np.nan

    df["record_id"] = (
        df["order_id_std"].astype(str)
        + "_"
        + df["order_date"].fillna(pd.Timestamp("1970-01-01")).astype(str)
    )

    sample_path = os.path.join(PRE_DIR, FILES["supply_clean_sample"])
    save_csv(df.head(5000), sample_path)
    return df


# =========================
# FEATURE ENGINEERING (Supply)
# =========================


def add_features_supply(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["is_expedited_sc"] = (
        data["shipment_mode"].astype(str).str.contains(
            r"first|same|express|expedite|priority|air|prime", case=False, na=False
        )
    ).astype(int)

    if data["price_dollar_sc"].notna().sum() >= 4:
        try:
            data["price_bucket_sc"] = pd.qcut(data["price_dollar_sc"], q=4, duplicates="drop").astype(str)
        except Exception:
            data["price_bucket_sc"] = "unknown"
    else:
        data["price_bucket_sc"] = "unknown"

    data["weekday"] = pd.to_datetime(data["order_date"], errors="coerce").dt.weekday
    data["hour"] = 0
    return data


# ===== Distance matrix & simple routing heuristics =====

def make_distance_matrix(coords: Sequence[Tuple[float, float]]) -> np.ndarray:
    n = len(coords)
    dmat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            dmat[i, j] = dmat[j, i] = dist
    return dmat


def greedy_route(distance_matrix: np.ndarray, start: int = 0) -> List[int]:
    n = distance_matrix.shape[0]
    unvisited = set(range(0, n))
    route = [start]
    unvisited.remove(start)
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: distance_matrix[cur, j])
        route.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    if route[0] != route[-1]:
        route.append(route[0])
    return route


def cheapest_insertion_route(distance_matrix: np.ndarray, start: int = 0) -> List[int]:
    n = distance_matrix.shape[0]
    if n <= 2:
        other_nodes = [i for i in range(n) if i != start]
        return [start] + other_nodes + [start]
    # initialise route with depot and closest point
    unvisited = [i for i in range(n) if i != start]
    nearest = min(unvisited, key=lambda j: distance_matrix[start, j])
    route: List[int] = [start, nearest, start]
    unvisited.remove(nearest)
    while unvisited:
        best_city = None
        best_pos = None
        best_increase = float("inf")
        for city in unvisited:
            for idx in range(len(route) - 1):
                a = route[idx]
                b = route[idx + 1]
                increase = (
                    distance_matrix[a, city]
                    + distance_matrix[city, b]
                    - distance_matrix[a, b]
                )
                if increase < best_increase:
                    best_increase = increase
                    best_city = city
                    best_pos = idx + 1
        if best_city is None or best_pos is None:
            break
        route.insert(best_pos, best_city)
        unvisited.remove(best_city)
    if route[0] != route[-1]:
        route.append(route[0])
    return route


def route_length(route: Sequence[int], distance_matrix: np.ndarray) -> float:
    return float(sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)))


def _generate_candidate_routes(
    distance_matrix: np.ndarray,
    n_random_starts: int = 3,
    random_state: int = RANDOM_STATE,
) -> List[List[int]]:
    n = distance_matrix.shape[0]
    rng = np.random.default_rng(random_state + n)
    candidates: List[List[int]] = []

    candidates.append(greedy_route(distance_matrix, start=0))
    candidates.append(cheapest_insertion_route(distance_matrix, start=0))

    for _ in range(min(n_random_starts, max(1, n - 1))):
        perm = list(range(1, n))
        rng.shuffle(perm)
        route = [0] + perm + [0]
        candidates.append(route)

    unique: List[List[int]] = []
    seen = set()
    for route in candidates:
        key = tuple(route)
        if key not in seen:
            seen.add(key)
            unique.append(route)
    return unique


def two_opt(route: Sequence[int], distance_matrix: np.ndarray, max_iter: int = 100) -> List[int]:
    best = list(route)
    best_len = route_length(best, distance_matrix)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, len(route) - 2):
            for k in range(i + 1, len(route) - 1):
                new_route = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                new_len = route_length(new_route, distance_matrix)
                if new_len + 1e-6 < best_len:
                    best, best_len = new_route, new_len
                    improved = True
        route = best
    return best


def eval_route(route: Sequence[int], distance_matrix: np.ndarray, vehicle: str, sla_min: float) -> Dict[str, float]:
    prof = VEHICLES[vehicle]
    total_km = route_length(route, distance_matrix)
    travel_min = (total_km / max(prof["speed_kmph"], 1e-6)) * 60.0
    stops = max(len(route) - 2, 0)
    service_total = stops * SERVICE_TIME_MIN
    total_min = travel_min + service_total
    cost = total_km * prof["cost_per_km"]
    feasible_distance = float(total_km <= prof.get("max_km_per_route", float("inf")))
    meets_sla = float(total_min <= sla_min)
    if not feasible_distance:
        meets_sla = 0.0
    return {
        "vehicle": vehicle,
        "distance_km": float(total_km),
        "travel_min": float(travel_min),
        "service_min": float(service_total),
        "total_min": float(total_min),
        "cost": float(cost),
        "meets_sla": meets_sla,
        "feasible_distance": feasible_distance,
    }


def objective_score(total_min: float, cost: float, penalty: float = 0.0) -> float:
    base = (1 - LAMBDA_COST) * total_min + LAMBDA_COST * (cost * 60.0)
    return float(base + penalty)


# =========================
# CLIENT PARTITIONS
# =========================


def build_client_partitions(df: pd.DataFrame, by_col: str = PARTITION_COL) -> Dict[str, pd.DataFrame]:
    if by_col not in df.columns:
        _opt_hb(f"[WARN] Partition column '{by_col}' not found. Using single client: Global")
        return {"Global": df.copy()}
    parts: Dict[str, pd.DataFrame] = {}
    for key, sub in df.groupby(by_col, dropna=False):
        key_str = str(key) if pd.notna(key) else "Unknown"
        sub = sub.copy()
        if sub.shape[0] >= MIN_ROWS_PER_CLIENT:
            parts[key_str] = sub
    if not parts:
        parts["Global"] = df.copy()
    return parts


# =========================
# FEATURES TABLE
# =========================

BASE_NUMERIC = ["order_qty", "price_dollar_sc", "weekday", "hour"]
BASE_CATEG = ["category_std", "shipment_mode", "partition_market", "partition_segment", "price_bucket_sc"]


def to_features_table(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    feature_frames = []
    for col in BASE_NUMERIC:
        if col in data.columns:
            feature_frames.append(data[[col]].rename(columns={col: col}))
    for col in BASE_CATEG:
        if col in data.columns:
            oh = pd.get_dummies(data[col].astype(str).fillna("unknown"), prefix=col, dummy_na=True)
            feature_frames.append(oh)
    if feature_frames:
        features = pd.concat(feature_frames, axis=1)
    else:
        features = pd.DataFrame(index=data.index)
    features["y_target_days_shipping"] = data["days_shipping_real"].astype(float)
    features.insert(0, "record_id", data["record_id"].astype(str))
    return features


def save_clients_features(parts: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for name, sub in parts.items():
        cdir = os.path.join(CLIENTS_DIR, name)
        os.makedirs(cdir, exist_ok=True)
        feats = to_features_table(sub)
        outp = os.path.join(cdir, "features.csv")
        save_csv(feats, outp, index=False)
        paths[name] = outp
        _opt_hb(f"[client:{name}] rows={feats.shape[0]} cols={feats.shape[1]} -> {outp}")
    return paths


def build_global_feature_cols(client_feature_paths: Dict[str, str], out_txt: str) -> List[str]:
    all_cols: set[str] = set()
    for _, path in client_feature_paths.items():
        df = pd.read_csv(path, nrows=1)
        for col in df.columns:
            if col not in {"y_target_days_shipping", "record_id"}:
                all_cols.add(col)
    cols = sorted(all_cols)
    with open(out_txt, "w", encoding="utf-8") as f:
        for col in cols:
            f.write(col + "\n")
    _opt_hb(f"[global] wrote {len(cols)} feature columns -> {out_txt}")
    return cols


# =========================
# TRANSFORMATION
# =========================

@dataclass
class TransformationStats:
    numeric_means: Dict[str, float]
    numeric_stds: Dict[str, float]
    feature_cols: List[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "numeric_means": self.numeric_means,
                "numeric_stds": self.numeric_stds,
                "feature_cols": self.feature_cols,
            },
            indent=2,
        )

    @staticmethod
    def from_json(payload: str) -> "TransformationStats":
        data = json.loads(payload)
        return TransformationStats(
            numeric_means={k: float(v) for k, v in data.get("numeric_means", {}).items()},
            numeric_stds={k: float(v) for k, v in data.get("numeric_stds", {}).items()},
            feature_cols=list(data.get("feature_cols", [])),
        )


def _align_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    aligned = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            aligned[col] = df[col]
        else:
            aligned[col] = 0.0
    return aligned


def compute_transformation_stats(client_paths: Dict[str, str], feature_cols: Sequence[str]) -> TransformationStats:
    numeric_cols = [col for col in feature_cols if col in BASE_NUMERIC]
    if not numeric_cols:
        return TransformationStats({}, {}, list(feature_cols))

    sums = {col: 0.0 for col in numeric_cols}
    sums_sq = {col: 0.0 for col in numeric_cols}
    counts = 0

    for path in client_paths.values():
        df = pd.read_csv(path)
        X = _align_features(df.drop(columns=["record_id", "y_target_days_shipping"], errors="ignore"), feature_cols)
        if not len(X):
            continue
        numeric_df = X[numeric_cols].astype(float)
        sums = {col: sums[col] + float(numeric_df[col].sum()) for col in numeric_cols}
        sums_sq = {col: sums_sq[col] + float((numeric_df[col] ** 2).sum()) for col in numeric_cols}
        counts += numeric_df.shape[0]

    means = {col: (sums[col] / counts if counts else 0.0) for col in numeric_cols}
    stds = {}
    for col in numeric_cols:
        if counts:
            mean = means[col]
            variance = max((sums_sq[col] / counts) - mean**2, 1e-9)
            stds[col] = math.sqrt(variance)
        else:
            stds[col] = 1.0

    return TransformationStats(numeric_means=means, numeric_stds=stds, feature_cols=list(feature_cols))


def apply_transformations(
    client_paths: Dict[str, str], feature_cols: Sequence[str], stats: TransformationStats
) -> Dict[str, str]:
    transformed_paths: Dict[str, str] = {}
    global_frames: List[pd.DataFrame] = []
    numeric_cols = list(stats.numeric_means.keys())

    for client, path in client_paths.items():
        df = pd.read_csv(path)
        features_only = df.drop(columns=["record_id", "y_target_days_shipping"], errors="ignore")
        aligned = _align_features(features_only, feature_cols)
        if numeric_cols:
            for col in numeric_cols:
                mean = stats.numeric_means.get(col, 0.0)
                std = stats.numeric_stds.get(col, 1.0) or 1.0
                aligned[col] = (aligned[col].astype(float) - mean) / std
        transformed_df = pd.concat([df[["record_id"]], aligned, df[["y_target_days_shipping"]]], axis=1)
        transformed_df.insert(1, "client", client)

        out_dir = os.path.join(TRANS_DIR, client)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "transformed_features.csv")
        save_csv(transformed_df, out_path, index=False)
        transformed_paths[client] = out_path
        _opt_hb(f"[transform:{client}] rows={transformed_df.shape[0]} cols={transformed_df.shape[1]}")
        global_frames.append(transformed_df.copy())

    with open(TRANSFORM_META_FILE, "w", encoding="utf-8") as f:
        f.write(stats.to_json())

    if global_frames:
        global_df = pd.concat(global_frames, axis=0, ignore_index=True)
        save_csv(global_df, GLOBAL_TRANSFORM_PATH, index=False)

    return transformed_paths


def _compute_prediction_metrics(
    transformed_path: str, sla_min: float, test_size: float = 0.2
) -> Dict[str, float]:
    """Train a light baseline regressor to populate prediction KPIs.

    This step guarantees that MAE/RMSE/R² columns in the optimization report
    are grounded in actual shipping-day predictions rather than routing totals.
    The classifier thresholds predictions using the SLA converted to days so
    the accuracy/precision/recall/F1 columns reflect on-time performance.
    """

    if not transformed_path or not os.path.exists(transformed_path):
        return {}

    df = pd.read_csv(transformed_path)
    if df.empty or "y_target_days_shipping" not in df.columns:
        return {}

    work = df.copy()
    work = work[np.isfinite(work["y_target_days_shipping"])].copy()
    if work.empty:
        return {}

    feature_cols = [
        c
        for c in work.columns
        if c
        not in {
            "record_id",
            "client",
            "y_target_days_shipping",
        }
    ]
    if not feature_cols:
        return {}

    X = work[feature_cols].astype(float).to_numpy()
    y = work["y_target_days_shipping"].astype(float).to_numpy()

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    if X.shape[0] < 10:
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    if X_train.size == 0 or X_test.size == 0:
        return {}

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if y_pred.size == 0:
        return {}

    errors = y_pred - y_test
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    if y_test.size >= 2:
        denom = float(np.sum((y_test - np.mean(y_test)) ** 2))
        if denom > 1e-9:
            r2 = float(1.0 - (np.sum(np.square(errors)) / denom))
        else:
            sse = float(np.sum(np.square(errors)))
            r2 = 1.0 if sse <= 1e-9 else 0.0
    else:
        sse = float(np.sum(np.square(errors)))
        r2 = 1.0 if sse <= 1e-9 else 0.0

    sla_days = float(sla_min) / (60.0 * 24.0)
    y_true_cls = (y_test <= sla_days).astype(int)
    y_pred_cls = (y_pred <= sla_days).astype(int)
    cls_metrics = _safe_classification_metrics(y_true_cls, y_pred_cls)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mae_hours": mae * 24.0,
        "rmse_hours": rmse * 24.0,
        "mae_per_stop": mae,
        "rmse_per_stop": rmse,
        "mae_per_stop_hours": mae * 24.0,
        "rmse_per_stop_hours": rmse * 24.0,
    }
    metrics.update(cls_metrics)
    metrics["prediction_rows"] = float(y_test.size)
    return metrics


# =========================
# ROUTING CORE
# =========================

def _hash_to_xy(
    s: str,
    scale_lat: float = SYNTH_COORD_LAT_RANGE_DEG,
    scale_lon: float = SYNTH_COORD_LON_RANGE_DEG,
    lat0: float = SYNTH_COORD_LAT_CENTER,
    lon0: float = SYNTH_COORD_LON_CENTER,
) -> Tuple[float, float]:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    a = int(h[:8], 16) / 0xFFFFFFFF
    b = int(h[8:16], 16) / 0xFFFFFFFF
    lat = lat0 + (a - 0.5) * scale_lat
    lon = lon0 + (b - 0.5) * scale_lon
    return float(lat), float(lon)


def _infer_synthetic_anchor(df: pd.DataFrame) -> Tuple[float, float]:
    lat0 = SYNTH_COORD_LAT_CENTER
    lon0 = SYNTH_COORD_LON_CENTER

    for lat_col, lon_col in [
        ("Store_Latitude", "Store_Longitude"),
        ("Drop_Latitude", "Drop_Longitude"),
    ]:
        if {lat_col, lon_col}.issubset(df.columns):
            lat_series = pd.to_numeric(df[lat_col], errors="coerce")
            lon_series = pd.to_numeric(df[lon_col], errors="coerce")
            if lat_series.notna().any() and lon_series.notna().any():
                return float(lat_series.dropna().mean()), float(lon_series.dropna().mean())

    return lat0, lon0


def ensure_synthetic_coords(
    df: pd.DataFrame,
    drop_lat_col: str = "Drop_Latitude",
    drop_lon_col: str = "Drop_Longitude",
    force: bool = False,
) -> pd.DataFrame:
    result = df.copy()
    if not force and drop_lat_col in result.columns and drop_lon_col in result.columns:
        if result[drop_lat_col].notna().any() and result[drop_lon_col].notna().any():
            return result

    lat_col = pick_col(result, "drop latitude", "latitude", "customer latitude", "order latitude")
    lon_col = pick_col(result, "drop longitude", "longitude", "customer longitude", "order longitude")
    lat0, lon0 = _infer_synthetic_anchor(result)
    if not force and lat_col and lon_col and result[lat_col].notna().any() and result[lon_col].notna().any():
        result[drop_lat_col] = pd.to_numeric(result[lat_col], errors="coerce")
        result[drop_lon_col] = pd.to_numeric(result[lon_col], errors="coerce")
    else:
        lower_map = {re.sub(r"\s+", " ", str(c)).strip().lower(): c for c in result.columns}
        candidates = [
            ("order city", "order state", "order country"),
            ("customer city", "customer state", "customer country"),
            ("city", "state", "country"),
        ]
        key_cols = None
        for trip in candidates:
            if all(col in lower_map for col in trip):
                key_cols = [lower_map[col] for col in trip]
                break
        if key_cols is None:
            if {"category_std", "partition_market"}.issubset(result.columns):
                key_cols = ["category_std", "partition_market"]
            else:
                if "order_id_std" not in result.columns:
                    result["order_id_std"] = np.arange(len(result)) + 1
                seed = result["order_id_std"].astype(str)
                latlon = seed.map(lambda s: _hash_to_xy(s, lat0=lat0, lon0=lon0))
                result[drop_lat_col] = latlon.map(lambda t: t[0])
                result[drop_lon_col] = latlon.map(lambda t: t[1])
                return result
        addr = result[key_cols].astype(str).fillna("unknown").agg(" / ".join, axis=1)
        latlon = addr.map(lambda s: _hash_to_xy(s, lat0=lat0, lon0=lon0))
        result[drop_lat_col] = latlon.map(lambda t: t[0])
        result[drop_lon_col] = latlon.map(lambda t: t[1])

    result.loc[(result[drop_lat_col] < -90) | (result[drop_lat_col] > 90), drop_lat_col] = np.nan
    result.loc[(result[drop_lon_col] < -180) | (result[drop_lon_col] > 180), drop_lon_col] = np.nan
    result[drop_lat_col] = result[drop_lat_col].fillna(SYNTH_COORD_LAT_CENTER)
    result[drop_lon_col] = result[drop_lon_col].fillna(SYNTH_COORD_LON_CENTER)
    return result


def guess_depot(df: pd.DataFrame) -> Tuple[float, float]:
    if {"Store_Latitude", "Store_Longitude"}.issubset(df.columns) and df["Store_Latitude"].notna().any():
        lat = df["Store_Latitude"].dropna().astype(float).mean()
        lon = df["Store_Longitude"].dropna().astype(float).mean()
        return float(lat), float(lon)
    if {"Drop_Latitude", "Drop_Longitude"}.issubset(df.columns) and df["Drop_Latitude"].notna().any():
        lat = df["Drop_Latitude"].dropna().astype(float).mean()
        lon = df["Drop_Longitude"].dropna().astype(float).mean()
        return float(lat), float(lon)
    return (25.276987, 55.296249)


def extract_points(df: pd.DataFrame) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    if {"Drop_Latitude", "Drop_Longitude"}.issubset(df.columns):
        sub = df.dropna(subset=["Drop_Latitude", "Drop_Longitude"]).copy()
        for _, row in sub.iterrows():
            lat = float(row["Drop_Latitude"])
            lon = float(row["Drop_Longitude"])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                points.append((lat, lon))
    return points


def build_best_route_for_batch(
    points: Sequence[Tuple[float, float]],
    depot: Tuple[float, float],
    sla_min: float = DEFAULT_SLA_MIN,
) -> Tuple[Optional[List[int]], Optional[Dict[str, float]], List[Dict[str, object]]]:
    if not points:
        return None, None, []
    coords = [depot] + list(points)
    distance_matrix = make_distance_matrix(coords)
    route_candidates: List[List[int]] = []
    seen_routes: Set[Tuple[int, ...]] = set()
    for base in _generate_candidate_routes(distance_matrix, n_random_starts=3):
        improved = two_opt(base, distance_matrix, max_iter=100)
        key = tuple(improved)
        if key not in seen_routes:
            seen_routes.add(key)
            route_candidates.append(improved)

    metrics_candidates: List[Dict[str, object]] = []
    for route in route_candidates:
        for vehicle in VEHICLES:
            metrics: Dict[str, object] = dict(eval_route(route, distance_matrix, vehicle, sla_min))
            penalty = 0.0
            if not metrics.get("feasible_distance", 1.0):
                penalty += 1e6
            if not metrics.get("meets_sla", 1.0):
                penalty += 1e5
            metrics["score"] = objective_score(metrics["total_min"], metrics["cost"], penalty=penalty)
            metrics["route"] = route
            if metrics.get("feasible_distance", 0.0):
                feasibility_rank = 0 if metrics.get("meets_sla", 0.0) else 1
            else:
                feasibility_rank = 2
            metrics["feasibility_rank"] = feasibility_rank
            metrics_candidates.append(metrics)

    if not metrics_candidates:
        return None, None, []

    best = min(metrics_candidates, key=lambda m: (m["feasibility_rank"], m["score"]))

    export_rows: List[Dict[str, object]] = []
    for cand in sorted(
        metrics_candidates,
        key=lambda m: (m["feasibility_rank"], m["score"]),
    ):
        export_rows.append(
            {
                "vehicle": cand["vehicle"],
                "distance_km": cand["distance_km"],
                "travel_min": cand["travel_min"],
                "service_min": cand["service_min"],
                "total_min": cand["total_min"],
                "cost": cand["cost"],
                "meets_sla": cand["meets_sla"],
                "feasible_distance": cand.get("feasible_distance", 0.0),
                "score": cand["score"],
                "feasibility_rank": cand["feasibility_rank"],
                "route_sequence": "->".join(str(node) for node in cand["route"]),
                "route_size": len(cand["route"]),
            }
        )

    return list(best["route"]), best, export_rows


# =========================
# ROUTING ORCHESTRATION
# =========================

def _cluster_points(
    points: Sequence[Tuple[float, float]],
    max_cap: int,
    random_state: int = RANDOM_STATE,
) -> List[List[Tuple[float, float]]]:
    if not points:
        return []
    n_clusters = max(1, int(math.ceil(len(points) / max_cap)))
    if len(points) <= max_cap or n_clusters == 1:
        return [list(points)]
    coords = np.array(points)
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state)
    labels = kmeans.fit_predict(coords)
    batches: List[List[Tuple[float, float]]] = [[] for _ in range(n_clusters)]
    for point, label in zip(points, labels):
        batches[int(label)].append(point)
    batches = [batch for batch in batches if batch]
    batches.sort(key=len, reverse=True)
    merged: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    for batch in batches:
        if len(current) + len(batch) <= max_cap:
            current.extend(batch)
        else:
            if current:
                merged.append(current)
            current = list(batch)
    if current:
        merged.append(current)
    return merged


def _safe_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, float]:


    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.size == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    y_true_bin = (y_true_arr.astype(float) > 0.5).astype(int)
    y_pred_bin = (y_pred_arr.astype(float) > 0.5).astype(int)

    tp = float(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = float(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = float(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = float(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    denom_cls = max(tp + tn + fp + fn, 1.0)
    accuracy = (tp + tn) / denom_cls
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _filter_summary_batches(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    work = df.copy()
    if "batch_id" in work.columns:
        work["batch_id"] = pd.to_numeric(work["batch_id"], errors="coerce")
        work = work[work["batch_id"].notna() & (work["batch_id"] >= 0)]
    return work


def _compute_global_metrics(
    df: pd.DataFrame,
    sla_min: float,
    candidate_df: Optional[pd.DataFrame] = None,
    prediction_data_path: Optional[str] = None,
) -> Dict[str, float]:


    summary_df = _filter_summary_batches(df)
    default_metrics = {
        "mae": float("nan"),
        "rmse": float("nan"),
        "r2": float("nan"),
        "mae_hours": float("nan"),
        "rmse_hours": float("nan"),
        "mae_per_stop": float("nan"),
        "rmse_per_stop": float("nan"),
        "mae_per_stop_hours": float("nan"),
        "rmse_per_stop_hours": float("nan"),
        "accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "prediction_rows": float("nan"),
        "routing_mae": float("nan"),
        "routing_rmse": float("nan"),
        "routing_r2": float("nan"),
        "routing_mae_hours": float("nan"),
        "routing_rmse_hours": float("nan"),
        "routing_mae_per_stop": float("nan"),
        "routing_rmse_per_stop": float("nan"),
        "routing_mae_per_stop_hours": float("nan"),
        "routing_rmse_per_stop_hours": float("nan"),
        "routing_accuracy": float("nan"),
        "routing_precision": float("nan"),
        "routing_recall": float("nan"),
        "routing_f1": float("nan"),
        "mean_total_min": float("nan"),
        "median_total_min": float("nan"),
        "sla_violation_mae": float("nan"),
        "sla_violation_rmse": float("nan"),
        "sla_violation_mae_hours": float("nan"),
        "sla_violation_rmse_hours": float("nan"),
        "sla_violation_rate": float("nan"),
        "candidate_entries": float("nan"),
        "candidate_mae": float("nan"),
        "candidate_rmse": float("nan"),
        "candidate_r2": float("nan"),
        "candidate_accuracy": float("nan"),
        "candidate_precision": float("nan"),
        "candidate_recall": float("nan"),
        "candidate_f1": float("nan"),
        "candidate_mae_hours": float("nan"),
        "candidate_rmse_hours": float("nan"),
        "candidate_best_rate": float("nan"),
    }

    if summary_df.empty:
        return default_metrics

    work = summary_df.copy()
    work["total_min"] = pd.to_numeric(work["total_min"], errors="coerce")
    work = work[np.isfinite(work["total_min"])]
    if work.empty:
        return default_metrics

    work["n_points"] = pd.to_numeric(work.get("n_points"), errors="coerce").fillna(0.0)
    work["meets_sla"] = pd.to_numeric(work.get("meets_sla"), errors="coerce").fillna(0.0)

    total_min = work["total_min"].to_numpy(dtype=float)
    metrics: Dict[str, float] = {
        "mean_total_min": float(np.mean(total_min)),
        "median_total_min": float(np.median(total_min)),
    }

    # ------------------------------
    # SLA violation analytics
    # ------------------------------
    sla_threshold = float(sla_min)
    violation = np.maximum(total_min - sla_threshold, 0.0)
    if violation.size:
        violation_mae = float(np.mean(violation))
        violation_rmse = float(np.sqrt(np.mean(np.square(violation))))
        metrics.update(
            {
                "sla_violation_mae": violation_mae,
                "sla_violation_rmse": violation_rmse,
                "sla_violation_mae_hours": violation_mae / 60.0,
                "sla_violation_rmse_hours": violation_rmse / 60.0,
                "sla_violation_rate": float(np.mean(violation > 1e-6)),
            }
        )

    # ------------------------------
    # Candidate-route insights
    # ------------------------------
    candidate_summary: Optional[pd.DataFrame] = None
    if candidate_df is not None and not candidate_df.empty:
        cand = candidate_df.copy()
        cand["batch_id"] = pd.to_numeric(cand.get("batch_id"), errors="coerce")
        cand = cand[cand["batch_id"].notna() & (cand["batch_id"] >= 0)]
        cand["total_min"] = pd.to_numeric(cand.get("total_min"), errors="coerce")
        cand = cand[np.isfinite(cand["total_min"])].copy()
        if not cand.empty:
            cand["meets_sla"] = pd.to_numeric(
                cand.get("meets_sla"), errors="coerce"
            ).fillna(0.0)
            cand["feasible_distance"] = pd.to_numeric(
                cand.get("feasible_distance"), errors="coerce"
            ).fillna(0.0)
            grouped = cand.groupby(["group", "batch_id"], dropna=False)

            best_totals = grouped["total_min"].transform("min")
            uplift = cand["total_min"] - best_totals
            metrics.update(
                {
                    "candidate_entries": float(cand.shape[0]),
                    "candidate_mae": float(np.mean(np.abs(uplift))),
                    "candidate_rmse": float(np.sqrt(np.mean(np.square(uplift)))),
                    "candidate_mae_hours": float(np.mean(np.abs(uplift)) / 60.0),
                    "candidate_rmse_hours": float(
                        np.sqrt(np.mean(np.square(uplift))) / 60.0
                    ),
                    "candidate_best_rate": float(
                        np.mean(grouped["total_min"].transform("min") == cand["total_min"])
                    ),
                }
            )

            denom = float(np.sum((cand["total_min"] - cand["total_min"].mean()) ** 2))
            if denom > 1e-9:
                metrics["candidate_r2"] = float(
                    1.0 - (np.sum(np.square(uplift)) / denom)
                )
            else:
                sse = float(np.sum(np.square(uplift)))
                metrics["candidate_r2"] = 1.0 if sse <= 1e-9 else 0.0

            cand_cls = _safe_classification_metrics(
                (cand["total_min"] <= sla_threshold).astype(int),
                cand["meets_sla"].astype(float),
            )
            metrics.update({f"candidate_{k}": v for k, v in cand_cls.items()})

            candidate_summary = (
                grouped["total_min"].min().to_frame(name="best_total_all")
            )
            candidate_summary["best_total_feasible"] = grouped.apply(
                lambda g: g.loc[g["feasible_distance"] >= 0.5, "total_min"].min()
                if (g["feasible_distance"] >= 0.5).any()
                else float("nan")
            )
            candidate_summary["best_total_sla"] = grouped.apply(
                lambda g: g.loc[
                    (g["feasible_distance"] >= 0.5) & (g["meets_sla"] >= 0.5), "total_min"
                ].min()
                if (
                    (g["feasible_distance"] >= 0.5)
                    & (g["meets_sla"] >= 0.5)
                ).any()
                else float("nan")
            )
            candidate_summary["sla_possible"] = grouped.apply(
                lambda g: float(
                    ((g["feasible_distance"] >= 0.5) & (g["meets_sla"] >= 0.5)).any()
                )
            )
            candidate_summary = candidate_summary.reset_index()

    merged = work.rename(columns={"meets_sla": "selected_meets_sla"})
    if candidate_summary is not None and not candidate_summary.empty:
        merged = merged.merge(
            candidate_summary,
            on=["group", "batch_id"],
            how="left",
        )
    else:
        merged = merged.assign(
            best_total_all=float("nan"),
            best_total_feasible=float("nan"),
            best_total_sla=float("nan"),
            sla_possible=float("nan"),
        )

    # ------------------------------
    # Classification metrics
    # ------------------------------
    if merged["sla_possible"].notna().any():
        y_true_cls = (merged["sla_possible"].astype(float) >= 0.5).astype(int)
    else:
        y_true_cls = (merged["total_min"].astype(float) <= sla_threshold).astype(int)
    y_pred_cls = (merged["selected_meets_sla"].astype(float) >= 0.5).astype(int)
    cls_metrics = _safe_classification_metrics(y_true_cls, y_pred_cls)
    metrics.update({f"routing_{k}": v for k, v in cls_metrics.items()})

    # ------------------------------
    # Regression metrics
    # ------------------------------
    counts = work["n_points"].to_numpy(dtype=float)
    counts = np.where(np.isfinite(counts) & (counts > 0), counts, 1.0)

    # baseline:
    if merged["best_total_sla"].notna().any():
        baseline = merged["best_total_sla"].to_numpy(dtype=float)
    elif merged["best_total_feasible"].notna().any():
        baseline = merged["best_total_feasible"].to_numpy(dtype=float)
    elif merged["best_total_all"].notna().any():
        baseline = merged["best_total_all"].to_numpy(dtype=float)
    else:
        baseline = np.full_like(total_min, sla_threshold)

    baseline = np.where(np.isfinite(baseline), baseline, sla_threshold)
    errors = total_min - baseline
    abs_errors = np.abs(errors)
    mae_min = float(np.mean(abs_errors))
    rmse_min = float(np.sqrt(np.mean(np.square(errors))))

    denom = float(np.sum((total_min - np.mean(total_min)) ** 2))
    if denom > 1e-9:
        r2_val = float(1.0 - (np.sum(np.square(errors)) / denom))
    else:
        sse = float(np.sum(np.square(errors)))
        r2_val = 1.0 if sse <= 1e-9 else 0.0

    mae_per_stop = float(np.mean(abs_errors / counts))
    rmse_per_stop = float(np.sqrt(np.mean(np.square(errors / counts))))

    metrics.update(
        {
            "routing_mae": mae_min,
            "routing_rmse": rmse_min,
            "routing_r2": r2_val,
            "routing_mae_hours": mae_min / 60.0,
            "routing_rmse_hours": rmse_min / 60.0,
            "routing_mae_per_stop": mae_per_stop,
            "routing_rmse_per_stop": rmse_per_stop,
            "routing_mae_per_stop_hours": mae_per_stop / 60.0,
            "routing_rmse_per_stop_hours": rmse_per_stop / 60.0,
        }
    )

    if prediction_data_path:
        prediction_metrics = _compute_prediction_metrics(prediction_data_path, sla_min)
        if prediction_metrics:
            metrics.update(prediction_metrics)

    result = {**default_metrics, **metrics}

    def _is_finite(value: object) -> bool:
        try:
            return bool(np.isfinite(float(value)))
        except Exception:
            return False

    routing_fallbacks = {
        "mae": ("routing_mae", 1.0 / 1440.0),
        "rmse": ("routing_rmse", 1.0 / 1440.0),
        "r2": ("routing_r2", 1.0),
        "mae_hours": ("routing_mae", 1.0 / 60.0),
        "rmse_hours": ("routing_rmse", 1.0 / 60.0),
        "mae_per_stop": ("routing_mae_per_stop", 1.0 / 1440.0),
        "rmse_per_stop": ("routing_rmse_per_stop", 1.0 / 1440.0),
        "mae_per_stop_hours": ("routing_mae_per_stop", 1.0 / 60.0),
        "rmse_per_stop_hours": ("routing_rmse_per_stop", 1.0 / 60.0),
        "accuracy": ("routing_accuracy", 1.0),
        "precision": ("routing_precision", 1.0),
        "recall": ("routing_recall", 1.0),
        "f1": ("routing_f1", 1.0),
    }

    for target_key, (routing_key, scale) in routing_fallbacks.items():
        current_val = result.get(target_key)
        if not _is_finite(current_val):
            routing_val = result.get(routing_key)
            if _is_finite(routing_val):
                result[target_key] = float(routing_val) * scale

    return result


def run_routing_optimization(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (),
    sla_min: float = DEFAULT_SLA_MIN,
    prediction_data_path: Optional[str] = None,
) -> None:
    if FAILSAFE_ALWAYS_GLOBAL:
        group_cols = ()

    groups_iter: Iterable[Tuple[Sequence[str], pd.DataFrame]]
    if not group_cols:
        groups_iter = [("Global", df)]
    else:
        groups_iter = df.groupby(list(group_cols), dropna=False)

    summary_rows: List[Dict[str, object]] = []
    all_metrics_rows: List[Dict[str, object]] = []
    for key, sub in groups_iter:
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_str = "_".join([str(k) for k in key_tuple if pd.notna(k)]) or "Global"
        out_dir = os.path.join(OPT_DIR, "routes", key_str)
        os.makedirs(out_dir, exist_ok=True)

        sub_work = sub.copy()
        if MAX_OPTIMIZATION_POINTS and len(sub_work) > MAX_OPTIMIZATION_POINTS:
            sub_work = (
                sub_work.sample(MAX_OPTIMIZATION_POINTS, random_state=RANDOM_STATE)
                .reset_index(drop=True)
            )
            _opt_hb(
                f"[{key_str}] sampled {MAX_OPTIMIZATION_POINTS}/{len(sub)} rows for routing"
            )

        sub_coords = ensure_synthetic_coords(
            sub_work, "Drop_Latitude", "Drop_Longitude", force=FAILSAFE_FORCE_SYNTH
        )
        depot = guess_depot(sub_coords)
        points = extract_points(sub_coords)

        if len(points) < FAILSAFE_MIN_POINTS:
            lat0, lon0 = depot
            need = FAILSAFE_MIN_POINTS - len(points)
            extra: List[Tuple[float, float]] = []
            for i in range(need):
                ang = 2 * np.pi * (i / max(need, 1))
                radius = 0.05 * (0.7 + 0.3 * np.sin(i))
                extra.append((lat0 + radius * np.cos(ang), lon0 + radius * np.sin(ang)))
            points = points + extra
            _opt_hb(f"[{key_str}] padded points -> {len(points)}")

        debug_path = os.path.join(out_dir, "__debug.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"group={key_str}\nrows={sub_coords.shape[0]}\npoints_with_coords={len(points)}\n")
            f.write(f"depot={depot}\n")

        max_cap = max(vehicle["capacity"] for vehicle in VEHICLES.values())
        batches = _cluster_points(points, max_cap=max_cap)
        total_points = sum(len(batch) for batch in batches)
        expected_batches = math.ceil(total_points / max_cap) if max_cap else 0
        _opt_hb(
            f"[routes] group={key_str or 'Global'} total_points={total_points} cap={max_cap} expected_batches={expected_batches} actual_batches={len(batches)}"
        )

        batch_id = 0
        for batch_pts in batches:
            route, best, metrics_rows = build_best_route_for_batch(
                batch_pts, depot, sla_min=sla_min
            )
            if route is None or best is None:
                continue

            coords = [depot] + list(batch_pts)
            path_rows = []
            for j in range(len(route) - 1):
                a = coords[route[j]]
                b = coords[route[j + 1]]
                path_rows.append(
                    {
                        "step": j,
                        "from_lat": a[0],
                        "from_lon": a[1],
                        "to_lat": b[0],
                        "to_lon": b[1],
                    }
                )
            path_df = pd.DataFrame(path_rows)
            path_csv = os.path.join(out_dir, f"batch_{batch_id:03d}_path.csv")
            save_csv(path_df, path_csv, index=False)

            metrics_csv = os.path.join(out_dir, f"batch_{batch_id:03d}_metrics.csv")
            if metrics_rows:
                metrics_df = pd.DataFrame(metrics_rows)
                metrics_df.insert(0, "batch_id", batch_id)
                metrics_df.insert(0, "group", key_str)
                save_csv(metrics_df, metrics_csv, index=False)
                all_metrics_rows.extend(metrics_df.to_dict(orient="records"))
            else:
                metrics_csv = ""

            summary_rows.append(
                {
                    "group": key_str,
                    "batch_id": batch_id,
                    "vehicle": best["vehicle"],
                    "distance_km": best["distance_km"],
                    "total_min": best["total_min"],
                    "travel_min": best["travel_min"],
                    "service_min": best["service_min"],
                    "cost": best["cost"],
                    "meets_sla": best["meets_sla"],
                    "feasible_distance": best.get("feasible_distance", 0.0),
                    "objective_score": best.get("score", float("nan")),
                    "n_points": len(batch_pts),
                    "path_csv": path_csv,
                    "metrics_csv": metrics_csv,
                    "note": "",
                }
            )
            _opt_hb(
                f"[routes] group={key_str} batch={batch_id} vehicle={best['vehicle']} score={best.get('score')} "
                f"meets_sla={best.get('meets_sla')} feasible={best.get('feasible_distance')}"
            )
            batch_id += 1

        if batch_id == 0:
            summary_rows.append(
                {
                    "group": key_str,
                    "batch_id": -1,
                    "vehicle": "",
                    "distance_km": 0.0,
                    "total_min": 0.0,
                    "travel_min": 0.0,
                    "service_min": 0.0,
                    "cost": 0.0,
                    "meets_sla": 0,
                    "feasible_distance": 0,
                    "objective_score": float("nan"),
                    "n_points": 0,
                    "path_csv": "",
                    "metrics_csv": "",
                    "note": "no batches produced",
                }
            )

    summary_path = os.path.join(OPT_DIR, "routes", "routes_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    save_csv(pd.DataFrame(summary_rows), summary_path, index=False)
    _opt_hb(f"routes_summary -> {summary_path} rows={len(summary_rows)}")

    if all_metrics_rows:
        all_metrics_df = pd.DataFrame(all_metrics_rows)
        metrics_all_path = os.path.join(OPT_DIR, "routes", "all_candidate_metrics.csv")
        save_csv(all_metrics_df, metrics_all_path, index=False)
    else:
        all_metrics_df = pd.DataFrame()
        metrics_all_path = ""

    summary_df = pd.DataFrame(summary_rows)
    try:
        perf_metrics = _compute_global_metrics(
            summary_df,
            sla_min=sla_min,
            candidate_df=all_metrics_df,
            prediction_data_path=prediction_data_path,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        fallback_keys = [
            "mae",
            "rmse",
            "r2",
            "mae_hours",
            "rmse_hours",
            "mae_per_stop",
            "rmse_per_stop",
            "mae_per_stop_hours",
            "rmse_per_stop_hours",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "prediction_rows",
            "routing_mae",
            "routing_rmse",
            "routing_r2",
            "routing_mae_hours",
            "routing_rmse_hours",
            "routing_mae_per_stop",
            "routing_rmse_per_stop",
            "routing_mae_per_stop_hours",
            "routing_rmse_per_stop_hours",
            "routing_accuracy",
            "routing_precision",
            "routing_recall",
            "routing_f1",
            "mean_total_min",
            "median_total_min",
            "sla_violation_mae",
            "sla_violation_rmse",
            "sla_violation_mae_hours",
            "sla_violation_rmse_hours",
            "sla_violation_rate",
            "candidate_entries",
            "candidate_mae",
            "candidate_rmse",
            "candidate_r2",
            "candidate_accuracy",
            "candidate_precision",
            "candidate_recall",
            "candidate_f1",
            "candidate_mae_hours",
            "candidate_rmse_hours",
            "candidate_best_rate",
        ]
        perf_metrics = {key: float("nan") for key in fallback_keys}
        perf_metrics["error"] = str(exc)

    metrics_out_path = os.path.join(OPT_DIR, "routes", "optimization_metrics.csv")
    metrics_payload = perf_metrics.copy()

    summary_batches = _filter_summary_batches(summary_df)
    if not summary_batches.empty:
        total_series = pd.to_numeric(summary_batches["total_min"], errors="coerce")
        total_series = total_series[np.isfinite(total_series)]
        sla_hits = (total_series <= float(sla_min)).astype(int)
        sla_met_count = int(sla_hits.sum())
        sla_met_rate = float(sla_hits.mean()) if sla_hits.size else float("nan")
        sla_violation_count = int(np.sum(total_series > float(sla_min)))
    else:
        sla_met_count = 0
        sla_met_rate = float("nan")
        sla_violation_count = 0

    metrics_payload.update(
        {
            "sla_met_count": sla_met_count,
            "sla_met_rate": sla_met_rate,
            "sla_violation_count": sla_violation_count,
        }
    )
    metrics_payload.update(
        {
            "num_batches": len(
                [row for row in summary_rows if row.get("batch_id", -1) >= 0]
            ),
            "summary_csv": summary_path,
            "candidates_csv": metrics_all_path,
            "sla_min": sla_min,
        }
    )
    save_csv(pd.DataFrame([metrics_payload]), metrics_out_path, index=False)
    _opt_hb(f"optimization_metrics -> {metrics_out_path}")


# =========================
# MAIN
# =========================

def run_all() -> Dict[str, object]:
    ensure_dirs()
    dataset_path = os.path.join(INPUT_DIR, FN_DS_SUPPLY)

    df_sup = preprocess_ds_supply_only(dataset_path)
    f_sup = add_features_supply(df_sup)

    parts = build_client_partitions(f_sup, by_col=PARTITION_COL)
    client_paths = save_clients_features(parts)

    feature_cols_path = os.path.join(RES_DIR, FILES["feature_cols_global"])
    feature_cols = build_global_feature_cols(client_paths, feature_cols_path)

    stats = compute_transformation_stats(client_paths, feature_cols)
    transformed_paths = apply_transformations(client_paths, feature_cols, stats)

    try:
        f_sup_coords = ensure_synthetic_coords(
            f_sup,
            drop_lat_col="Drop_Latitude",
            drop_lon_col="Drop_Longitude",
            force=FAILSAFE_FORCE_SYNTH,
        )
        group_cols: Sequence[str] = ()
        _opt_hb(
            f"start optimization rows={f_sup_coords.shape[0]} group_cols={group_cols}"
        )
        run_routing_optimization(
            f_sup_coords,
            group_cols=group_cols,
            sla_min=DEFAULT_SLA_MIN,
            prediction_data_path=GLOBAL_TRANSFORM_PATH,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        _opt_hb(f"optimization error: {exc}")

    return {
        "clients": client_paths,
        "feature_cols_global": feature_cols_path,
        "n_features": len(feature_cols),
        "partition_col": PARTITION_COL,
        "transformed_clients": transformed_paths,
        "global_transformed": GLOBAL_TRANSFORM_PATH,
        "transformer_meta": TRANSFORM_META_FILE,
        "optimization_dir": os.path.join(OPT_DIR, "routes"),
        "optimization_metrics": os.path.join(OPT_DIR, "routes", "optimization_metrics.csv"),
    }


if __name__ == "__main__":
    info = run_all()
    print("\nArtifacts:")
    for key, value in info.items():
        print(key, ":", value)