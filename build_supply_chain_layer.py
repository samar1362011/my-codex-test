# build_supply_chain_layer.py
# -----------------------------------------------------------------------------
# Independent "Supply Chain" layer generator for the existing project.
#
# What this script does (no changes to your current pipeline):
# - Reads your dataset (prefers DATASET/DS_DataCoSupplyChainDataset.csv; falls
#   back to RES/preprocessing/DS_SupplyChain_cleaned_sample.csv if present).
# - Infers products, categories, markets/countries, and order records.
# - Synthesizes a *separate* supply-chain layer (CSV files) with filled rows:
#     * suppliers.csv
#     * warehouses.csv
#     * products.csv
#     * bom.csv
#     * production_batches.csv
#     * shipments.csv
#     * trips.csv
#     * sla.csv
#     * order_to_shipment.csv
#     * shipment_to_trip.csv
#     * batch_to_shipment.csv
#     * customers.csv
#     * dim_locations.csv
#     * plants.csv
#     * orders_enriched.csv
# - All artifacts are written to: RES/supply_chain_layer/
#
# You can run this script in PyCharm:
#   python build_supply_chain_layer.py
#
# Notes
# - The layer is independent: it does NOT overwrite anything in your original
#   dataset or preprocessing outputs. You can integrate it later.
# - Row counts scale with your dataset (orders) to remain “appropriate” in size.
# - Pure pandas/numpy. No external libs needed.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import math
import json
import hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ------------------------------- Config --------------------------------------

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.getcwd()
DATASET_PRIMARY = os.path.join(BASE_DIR, "DATASET", "DS_DataCoSupplyChainDataset.csv")
DATASET_FALLBACK = os.path.join(BASE_DIR, "RES", "preprocessing", "DS_SupplyChain_cleaned_sample.csv")

OUT_DIR = os.path.join(BASE_DIR, "RES", "supply_chain_layer")
os.makedirs(OUT_DIR, exist_ok=True)

# Targets for scaling (heuristics)
# - shipments: roughly group small sets of orders together
# - trips: group shipments per warehouse/day
ORDERS_PER_SHIPMENT = 8             # avg orders per shipment group
SHIPMENTS_PER_TRIP = 10             # avg shipments per trip
BATCH_SIZE_UNITS = 200              # avg units per production batch (scaled per product)
BOM_MIN_COMPONENTS = 1
BOM_MAX_COMPONENTS = 3

# If your dataset lacks lat/lon, we derive synthetic ones in a reproducible way
SYNTH_LAT_CENTER = 25.276987
SYNTH_LON_CENTER = 55.296249
SYNTH_LAT_RANGE = 0.8
SYNTH_LON_RANGE = 1.0

VEHICLE_PROFILES = {
    "bike": {"speed_kmph": 18.0, "cost_per_km": 0.35, "capacity_units": 400},
    "car": {"speed_kmph": 45.0, "cost_per_km": 0.55, "capacity_units": 1800},
    "van": {"speed_kmph": 60.0, "cost_per_km": 0.75, "capacity_units": 3200},
    "truck": {"speed_kmph": 70.0, "cost_per_km": 0.9, "capacity_units": 6000},
}

CUSTOMER_TIERS = ["bronze", "silver", "gold", "platinum"]
CUSTOMER_SEGMENTS = ["retail", "wholesale", "hospitality", "corporate"]
SUPPLIER_LOCATIONS = [
    ("Dubai", "United Arab Emirates"),
    ("Riyadh", "Saudi Arabia"),
    ("Doha", "Qatar"),
    ("Manama", "Bahrain"),
    ("Kuwait City", "Kuwait"),
]
WAREHOUSE_LOCATIONS = [
    ("Dubai", "United Arab Emirates"),
    ("Abu Dhabi", "United Arab Emirates"),
    ("Sharjah", "United Arab Emirates"),
    ("Jeddah", "Saudi Arabia"),
    ("Muscat", "Oman"),
]

PLANT_LOCATIONS = [
    ("Jebel Ali", "United Arab Emirates"),
    ("Ras Al Khaimah", "United Arab Emirates"),
    ("Dammam", "Saudi Arabia"),
    ("Fujairah", "United Arab Emirates"),
    ("Salalah", "Oman"),
]

SLA_OPTIONS = {
    "SameDay": 12 * 60,
    "NextDay": 24 * 60,
    "TwoDay": 48 * 60,
    "ThreeDay": 72 * 60,
}





# ----------------------------- IO Utilities ----------------------------------

def read_csv_robust(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except Exception:
            continue
    return None


def write_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# ----------------------------- Helpers ---------------------------------------

def pick_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in cands:
        key = str(cand).strip().lower()
        if key in low:
            return low[key]
    return None


def normalize_text(s: object) -> str:
    if pd.isna(s):
        return ""
    return " ".join(str(s).strip().split())


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def synth_xy(seed: str) -> Tuple[float, float]:
    h = hashlib.md5(seed.encode("utf-8")).hexdigest()
    a = int(h[:8], 16) / 0xFFFFFFFF
    b = int(h[8:16], 16) / 0xFFFFFFFF
    lat = SYNTH_LAT_CENTER + (a - 0.5) * SYNTH_LAT_RANGE
    lon = SYNTH_LON_CENTER + (b - 0.5) * SYNTH_LON_RANGE
    return float(lat), float(lon)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2 * math.asin(math.sqrt(max(a, 0.0)))
    return 6371.0 * c


# -------------------------- Source Data Loading -------------------------------

def load_source_orders() -> pd.DataFrame:
    # Prefer the primary dataset, else fallback to preprocessed sample
    df = read_csv_robust(DATASET_PRIMARY)
    if df is None:
        df = read_csv_robust(DATASET_FALLBACK)
    if df is None:
        raise FileNotFoundError(
            "No dataset found. Ensure DATASET/DS_DataCoSupplyChainDataset.csv "
            "or RES/preprocessing/DS_SupplyChain_cleaned_sample.csv exists."
        )

    # Try to discover key columns in a tolerant way
    col_order_id = pick_col(df, "order id", "order item id", "orderitem id", "order number", "order_number")
    col_order_date = pick_col(df, "order date", "order created date", "order purchased date", "order date (dateorders)")
    col_ship_date = pick_col(df, "shipping date", "shipping date (date orders)", "ship date", "ship date (dateorders)")
    col_qty = pick_col(df, "order item quantity", "order quantity", "quantity")
    col_price = pick_col(df, "order item product price", "product price", "unit price", "item price", "order item total")
    col_prod = pick_col(df, "product name", "product", "product_name")
    col_cat = pick_col(df, "category name", "product category", "category", "product type", "category_std")
    col_market = pick_col(df, "market", "region", "customer market")
    col_city = pick_col(df, "order city", "customer city", "city")
    col_state = pick_col(df, "order state", "customer state", "state")
    col_country = pick_col(df, "order country", "customer country", "country")
    col_lat = pick_col(df, "drop latitude", "latitude", "customer latitude", "order latitude")
    col_lon = pick_col(df, "drop longitude", "longitude", "customer longitude", "order longitude")

    work = pd.DataFrame()
    work["record_id"] = (df[col_order_id].astype(str) if col_order_id else (np.arange(len(df)) + 1).astype(str))
    work["order_date"] = to_datetime(df[col_order_date]) if col_order_date else pd.NaT
    work["ship_date"] = to_datetime(df[col_ship_date]) if col_ship_date else pd.NaT
    work["qty"] = pd.to_numeric(df[col_qty], errors="coerce") if col_qty else np.nan
    work["price"] = pd.to_numeric(df[col_price], errors="coerce") if col_price else np.nan
    work["product_name"] = df[col_prod].map(normalize_text) if col_prod else "Unknown Product"
    work["category_std"] = df[col_cat].map(normalize_text) if col_cat else "unknown"
    work["market"] = df[col_market].map(normalize_text) if col_market else "Global"
    work["order_city"] = df[col_city].map(normalize_text) if col_city else ""
    work["order_state"] = df[col_state].map(normalize_text) if col_state else ""
    work["order_country"] = df[col_country].map(normalize_text) if col_country else ""

    # Locations (real or synthetic)
    if col_lat and col_lon:
        lat = pd.to_numeric(df[col_lat], errors="coerce")
        lon = pd.to_numeric(df[col_lon], errors="coerce")
        if lat.notna().any() and lon.notna().any():
            work["drop_lat"] = lat
            work["drop_lon"] = lon
        else:
            latlon = work["record_id"].map(lambda s: synth_xy(s))
            work["drop_lat"] = latlon.map(lambda t: t[0])
            work["drop_lon"] = latlon.map(lambda t: t[1])
    else:
        latlon = work["record_id"].map(lambda s: synth_xy(s))
        work["drop_lat"] = latlon.map(lambda t: t[0])
        work["drop_lon"] = latlon.map(lambda t: t[1])

    # Minimal clean-up
    work["qty"] = work["qty"].fillna(1).clip(lower=1).astype(int)
    work["price"] = work["price"].fillna(work["price"].median() if pd.notna(work["price"].median()) else 50.0)

    # Temporal fallbacks
    if work["order_date"].isna().all():
        # derive pseudo time series using hash order
        rng = np.random.default_rng(RANDOM_STATE)
        t0 = pd.Timestamp("2016-01-01")
        offsets = rng.integers(0, 365, size=len(work))
        work["order_date"] = t0 + pd.to_timedelta(offsets, unit="D")
    if work["ship_date"].isna().all():
        work["ship_date"] = work["order_date"] + pd.to_timedelta(np.random.randint(0, 7, size=len(work)), unit="D")

    # Add week buckets for grouping shipments
    work["week_start"] = work["order_date"].dt.to_period("W").apply(lambda r: r.start_time)

    return work


def _choose_customer_tier(series: pd.Series) -> str:
    if series <= 2:
        return CUSTOMER_TIERS[0]
    if series <= 5:
        return CUSTOMER_TIERS[1]
    if series <= 10:
        return CUSTOMER_TIERS[2]
    return CUSTOMER_TIERS[3]


def make_customers(orders: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = orders.copy()
    cols = ["order_country", "order_state", "order_city", "market"]
    for col in cols:
        base[col] = base[col].fillna("").astype(str)

    grouped = (
        base.groupby(cols)
        .agg({
            "record_id": "count",
            "qty": "sum",
            "drop_lat": "mean",
            "drop_lon": "mean",
        })
        .reset_index()
    )

    grouped["customer_index"] = np.arange(len(grouped))
    grouped["customer_id"] = grouped["customer_index"].apply(lambda x: f"CUST{x+1:06d}")
    rng = np.random.default_rng(RANDOM_STATE)
    grouped["customer_name"] = grouped.apply(
        lambda r: f"{r['order_city'] or 'Customer'} {int(r['customer_index'])+1}", axis=1
    )
    grouped["customer_segment"] = rng.choice(CUSTOMER_SEGMENTS, size=len(grouped))
    grouped["customer_tier"] = grouped["qty"].map(_choose_customer_tier)

    grouped["drop_lat"] = grouped["drop_lat"].fillna(0.0)
    grouped["drop_lon"] = grouped["drop_lon"].fillna(0.0)
    missing_lat = grouped["drop_lat"] == 0.0
    if missing_lat.any():
        coords = grouped.loc[missing_lat, "customer_id"].apply(synth_xy)
        grouped.loc[missing_lat, "drop_lat"] = coords.apply(lambda t: float(t[0])).values
        grouped.loc[missing_lat, "drop_lon"] = coords.apply(lambda t: float(t[1])).values

    customers = grouped[[
        "customer_id",
        "customer_name",
        "customer_segment",
        "customer_tier",
        "market",
        "order_country",
        "order_state",
        "order_city",
        "drop_lat",
        "drop_lon",
    ]].rename(
        columns={
            "order_country": "country",
            "order_state": "state",
            "order_city": "city",
            "drop_lat": "latitude",
            "drop_lon": "longitude",
        }
    )
    customers["location_id"] = customers["customer_id"].map(lambda cid: f"LOC_CUST_{cid[4:]}")

    join_cols = ["order_country", "order_state", "order_city", "market"]
    orders = orders.merge(
        customers[["customer_id", "country", "state", "city", "market"]]
        .rename(columns={"country": "order_country", "state": "order_state", "city": "order_city"}),
        on=join_cols,
        how="left",
    )

    orders["customer_id"] = orders["customer_id"].fillna("CUST000000")
    orders["customer_location_id"] = orders["customer_id"].map(lambda cid: f"LOC_CUST_{cid[4:]}" if cid != "CUST000000" else "LOC_CUST_000000")

    return customers, orders


def augment_orders_with_schedule(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return orders

    orders = orders.copy()
    rng = np.random.default_rng(RANDOM_STATE)

    order_dates = pd.to_datetime(orders["order_date"], errors="coerce")
    ship_dates = pd.to_datetime(orders["ship_date"], errors="coerce")
    fallback = pd.Timestamp("2016-01-01")

    requested_offsets = rng.integers(1, 3, size=len(orders))
    promised_offsets = rng.integers(1, 3, size=len(orders))
    actual_offsets = rng.integers(0, 4, size=len(orders))

    requested = (order_dates.fillna(fallback) + pd.to_timedelta(requested_offsets, unit="D")).dt.floor("H")
    promised = (requested + pd.to_timedelta(promised_offsets, unit="D")).dt.floor("H")
    actual_base = ship_dates.fillna(order_dates.fillna(fallback))
    actual = (actual_base + pd.to_timedelta(actual_offsets, unit="D")).dt.floor("H")

    sla_keys = rng.choice(list(SLA_OPTIONS.keys()), size=len(orders), p=[0.2, 0.45, 0.25, 0.1])
    sla_minutes = pd.Series(sla_keys).map(SLA_OPTIONS).astype(float)

    delay_hours = (actual - promised).dt.total_seconds() / 3600.0
    orders["requested_delivery_date"] = requested
    orders["promised_delivery_date"] = promised
    orders["actual_delivery_date"] = actual
    orders["sla_name"] = sla_keys
    orders["sla_minutes"] = sla_minutes
    orders["delivery_delay_hours"] = delay_hours.fillna(0.0)
    orders["on_time_flag"] = (orders["delivery_delay_hours"] <= 0.01).astype(int)

    return orders


# ---------------------- Synthetic Entity Generation --------------------------

def make_products(orders: pd.DataFrame) -> pd.DataFrame:
    # Use unique product names present; keep category if available
    prods = orders[["product_name", "category_std"]].dropna().drop_duplicates().copy()
    # Generate product_id
    prods = prods.sort_values(["category_std", "product_name"]).reset_index(drop=True)
    prods["product_id"] = ["P%06d" % (i + 1) for i in range(len(prods))]

    # Simple weights/volumes from price and qty stats
    rng = np.random.default_rng(RANDOM_STATE)
    base_w = np.clip(orders["price"].fillna(50) / 50.0, 0.2, 8.0)
    # Map product → typical weight/volume
    w_lookup = orders.groupby("product_name")["price"].median().to_dict()
    def weight_for(p):
        v = w_lookup.get(p, 50.0)
        rng_jitter = rng.uniform(0.8, 1.2)
        return float(np.clip(v / 50.0 * rng_jitter, 0.2, 10.0))
    prods["unit_weight_kg"] = prods["product_name"].map(weight_for).round(2)
    prods["unit_volume_l"] = (prods["unit_weight_kg"] * 0.8 * rng.uniform(0.6, 1.2, size=len(prods))).round(2)

    # Reorder columns
    prods = prods[["product_id", "product_name", "category_std", "unit_weight_kg", "unit_volume_l"]]
    return prods


def make_suppliers(products: pd.DataFrame) -> pd.DataFrame:
    # One supplier per category (plus some extras if categories are few)
    cats = products["category_std"].fillna("unknown").unique().tolist()
    rng = np.random.default_rng(RANDOM_STATE)
    n_extra = max(0, 3 - len(cats))
    cat_list = cats + [f"virtual_cat_{i+1}" for i in range(n_extra)]

    rows = []
    for i, cat in enumerate(sorted(cat_list)):
        lead = int(np.clip(2 + i % 5, 1, 10))
        cap = int(500 + (i * 75))
        city, country = SUPPLIER_LOCATIONS[i % len(SUPPLIER_LOCATIONS)]
        lat, lon = synth_xy(f"supplier_{cat}_{i}")
        rows.append({
            "supplier_id": f"S{1000+i}",
            "supplier_name": f"{cat.title()} Supplies Co.",
            "region": f"Region-{(i % 5) + 1}",
            "country": country,
            "city": city,
            "latitude": float(lat),
            "longitude": float(lon),
            "lead_time_days": lead,
            "max_daily_capacity_units": cap,
        })
    suppliers = pd.DataFrame(rows)
    suppliers["location_id"] = suppliers["supplier_id"].map(lambda sid: f"LOC_SUP_{sid[1:]}")
    return suppliers


def make_warehouses(orders: pd.DataFrame) -> pd.DataFrame:
    # Use country and/or market to define hubs
    markets = orders["market"].replace("", "Global").fillna("Global")
    countries = orders["order_country"].replace("", "Unknown").fillna("Unknown")

    hubs = (
        pd.DataFrame({"market": markets, "order_country": countries})
        .value_counts()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values("count", ascending=False)
    )
    # Cap the number of warehouses (sufficient coverage)
    n_wh = int(np.clip(len(hubs), 3, 20))
    hubs = hubs.head(n_wh).reset_index(drop=True)

    rows = []
    for i, row in hubs.iterrows():
        w_id = f"W{100+i}"
        name = f"{row['market'] or 'Global'}-{row['order_country'] or 'Unknown'} Hub"
        cap = int(5000 + (i * 500))
        handling = 2 + (i % 3)  # hours
        city, country = WAREHOUSE_LOCATIONS[i % len(WAREHOUSE_LOCATIONS)]
        lat, lon = synth_xy(f"warehouse_{w_id}_{city}")
        rows.append({
            "warehouse_id": w_id,
            "warehouse_name": name,
            "region": row["market"] or "Global",
            "country": country,
            "city": city,
            "latitude": float(lat),
            "longitude": float(lon),
            "capacity_units": cap,
            "handling_time_hours": handling,
        })
    warehouses = pd.DataFrame(rows)
    warehouses["location_id"] = warehouses["warehouse_id"].map(lambda wid: f"LOC_WH_{wid[1:]}")
    return warehouses


def make_bom(products: pd.DataFrame) -> pd.DataFrame:
    # Create 1–3 simple components per product (synthetic)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for _, p in products.iterrows():
        k = rng.integers(BOM_MIN_COMPONENTS, BOM_MAX_COMPONENTS + 1)
        for j in range(int(k)):
            rows.append({
                "product_id": p["product_id"],
                "component_id": f"C{p['product_id'][1:]}_{j+1:02d}",
                "component_name": f"Component-{j+1}",
                "component_type": "raw" if j < k - 1 else "packaging",
                "qty_per_unit": int(rng.integers(1, 4)),
            })
    return pd.DataFrame(rows)


def make_dim_locations(
    customers: pd.DataFrame,
    suppliers: pd.DataFrame,
    warehouses: pd.DataFrame,
    plant_entries: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    frames = []

    if not customers.empty:
        cust = customers.copy()
        cust["location_id"] = cust["customer_id"].map(lambda cid: f"LOC_CUST_{cid[4:]}")
        frames.append(
            cust[
                [
                    "location_id",
                    "customer_id",
                    "customer_name",
                    "customer_segment",
                    "customer_tier",
                    "country",
                    "state",
                    "city",
                    "latitude",
                    "longitude",
                ]
            ]
            .rename(
                columns={
                    "customer_id": "entity_id",
                    "customer_name": "entity_name",
                }
            )
            .assign(entity_type="customer")
        )

    if not suppliers.empty:
        frames.append(
            suppliers[[
                "location_id",
                "supplier_id",
                "supplier_name",
                "country",
                "city",
                "latitude",
                "longitude",
            ]]
            .rename(
                columns={
                    "supplier_id": "entity_id",
                    "supplier_name": "entity_name",
                }
            )
            .assign(entity_type="supplier", state="")
        )

    if not warehouses.empty:
        frames.append(
            warehouses[[
                "location_id",
                "warehouse_id",
                "warehouse_name",
                "country",
                "city",
                "latitude",
                "longitude",
            ]]
            .rename(
                columns={
                    "warehouse_id": "entity_id",
                    "warehouse_name": "entity_name",
                }
            )
            .assign(entity_type="warehouse", state="")
        )

    if plant_entries is not None and not plant_entries.empty:
        frames.append(
            plant_entries[[
                "location_id",
                "plant_id",
                "plant_name",
                "country",
                "city",
                "latitude",
                "longitude",
            ]]
            .rename(
                columns={
                    "plant_id": "entity_id",
                    "plant_name": "entity_name",
                }
            )
            .assign(entity_type="plant", state="")
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "location_id",
                "entity_id",
                "entity_name",
                "entity_type",
                "country",
                "state",
                "city",
                "latitude",
                "longitude",
            ]
        )

    dim_locations = pd.concat(frames, ignore_index=True)
    dim_locations["latitude"] = pd.to_numeric(dim_locations["latitude"], errors="coerce").fillna(0.0)
    dim_locations["longitude"] = pd.to_numeric(dim_locations["longitude"], errors="coerce").fillna(0.0)
    missing = (dim_locations["latitude"] == 0.0) | (dim_locations["longitude"] == 0.0)
    if missing.any():
        coords = dim_locations.loc[missing, "location_id"].apply(synth_xy)
        dim_locations.loc[missing, "latitude"] = coords.apply(lambda t: float(t[0])).values
        dim_locations.loc[missing, "longitude"] = coords.apply(lambda t: float(t[1])).values

    return dim_locations


def assign_warehouses_to_orders(orders: pd.DataFrame, warehouses: pd.DataFrame) -> pd.Series:
    # Map each order to the best matching warehouse by market/country (fallback to round-robin)
    if warehouses.empty:
        return pd.Series(["W100"] * len(orders), index=orders.index)

    # Create a key for join
    orders_key = (orders["market"].replace("", "Global").fillna("Global") + "||" +
                  orders["order_country"].replace("", "Unknown").fillna("Unknown"))
    wh_key = warehouses["region"].fillna("Global") + "||" + warehouses["warehouse_name"].str.extract(r"-(.*) Hub", expand=False).fillna("Unknown")
    wh_map = dict(zip(wh_key, warehouses["warehouse_id"]))

    assigned = []
    wh_ids = warehouses["warehouse_id"].tolist()
    for i, key in enumerate(orders_key):
        assigned.append(wh_map.get(key, wh_ids[i % len(wh_ids)]))
    return pd.Series(assigned, index=orders.index)


def make_shipments(
    orders: pd.DataFrame,
    warehouses: pd.DataFrame,
    customers: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if orders.empty:
        return pd.DataFrame(), pd.DataFrame()

    wh_for_order = assign_warehouses_to_orders(orders, warehouses)
    orders = orders.copy()
    orders["warehouse_id"] = wh_for_order.values

    warehouse_lookup = warehouses.set_index("warehouse_id")
    customer_lookup = customers.set_index("customer_id") if not customers.empty else pd.DataFrame()

    frame = orders[[
        "record_id",
        "warehouse_id",
        "customer_id",
        "customer_location_id",
        "order_city",
        "order_state",
        "order_country",
        "week_start",
        "qty",
        "requested_delivery_date",
        "promised_delivery_date",
        "actual_delivery_date",
        "delivery_delay_hours",
        "sla_name",
        "sla_minutes",
        "drop_lat",
        "drop_lon",
    ]].copy()

    frame["customer_id"] = frame["customer_id"].fillna("CUST000000")
    frame["customer_location_id"] = frame["customer_location_id"].fillna("LOC_CUST_000000")

    frame = frame.sort_values([
        "warehouse_id",
        "customer_id",
        "week_start",
        "requested_delivery_date",
    ])

    rows: List[Dict] = []
    rel_rows: List[Dict] = []
    shipment_counter = 0

    def chunk_iterable(idx_list: List[int], chunk_size: int):
        for i in range(0, len(idx_list), chunk_size):
            yield idx_list[i : i + chunk_size]

    group_cols = ["warehouse_id", "customer_id", "week_start"]
    for key, sub in frame.groupby(group_cols, dropna=False):
        idxs = sub.index.tolist()
        for chunk in chunk_iterable(idxs, ORDERS_PER_SHIPMENT):
            shipment_id = f"SH{shipment_counter:07d}"
            shipment_counter += 1
            sub_chunk = frame.loc[chunk]
            total_qty = int(sub_chunk["qty"].sum())
            requested = pd.to_datetime(sub_chunk["requested_delivery_date"]).min()
            promised = pd.to_datetime(sub_chunk["promised_delivery_date"]).max()
            actual = pd.to_datetime(sub_chunk["actual_delivery_date"]).max()
            sla_minutes = float(sub_chunk["sla_minutes"].max()) if sub_chunk["sla_minutes"].notna().any() else 24 * 60
            delay_hours = (actual - promised).total_seconds() / 3600.0 if pd.notna(actual) and pd.notna(promised) else 0.0
            on_time = 1 if delay_hours <= 0.01 else 0

            warehouse_id, customer_id, week_start = key
            warehouse_row = warehouse_lookup.loc[warehouse_id] if warehouse_id in warehouse_lookup.index else None
            customer_row = customer_lookup.loc[customer_id] if not customer_lookup.empty and customer_id in customer_lookup.index else None

            if warehouse_row is not None:
                origin_loc = warehouse_row.get("location_id", "LOC_WH_UNKNOWN")
                origin_lat = float(warehouse_row.get("latitude", SYNTH_LAT_CENTER))
                origin_lon = float(warehouse_row.get("longitude", SYNTH_LON_CENTER))
            else:
                origin_loc = "LOC_WH_UNKNOWN"
                origin_lat, origin_lon = synth_xy(warehouse_id or "warehouse")

            if customer_row is not None:
                dest_loc = f"LOC_CUST_{customer_id[4:]}"
                dest_lat = float(customer_row.get("latitude", SYNTH_LAT_CENTER))
                dest_lon = float(customer_row.get("longitude", SYNTH_LON_CENTER))
            else:
                dest_loc = sub_chunk["customer_location_id"].iloc[0]
                dest_lat = float(sub_chunk["drop_lat"].mean())
                dest_lon = float(sub_chunk["drop_lon"].mean())
                if math.isclose(dest_lat, 0.0) and math.isclose(dest_lon, 0.0):
                    dest_lat, dest_lon = synth_xy(dest_loc)

            distance_km = float(haversine_km(origin_lat, origin_lon, dest_lat, dest_lon))
            profile = VEHICLE_PROFILES["van" if total_qty > 1000 else "car"]
            planned_hours = distance_km / max(profile["speed_kmph"], 1e-3)
            actual_hours = planned_hours + max(delay_hours, 0.0)
            base_cost = distance_km * profile["cost_per_km"]

            rows.append({
                "shipment_id": shipment_id,
                "warehouse_id": warehouse_id,
                "customer_id": customer_id,
                "origin_location_id": origin_loc,
                "destination_location_id": dest_loc,
                "week_start": week_start,
                "orders_count": int(len(sub_chunk)),
                "total_units": total_qty,
                "requested_delivery_date": requested.isoformat() if pd.notna(requested) else "",
                "promised_delivery_date": promised.isoformat() if pd.notna(promised) else "",
                "actual_delivery_date": actual.isoformat() if pd.notna(actual) else "",
                "delivery_delay_hours": float(delay_hours),
                "on_time_flag": int(on_time),
                "sla_name": sub_chunk["sla_name"].mode().iat[0] if not sub_chunk["sla_name"].mode().empty else "NextDay",
                "sla_minutes": sla_minutes,
                "distance_km": distance_km,
                "planned_transit_hours": float(planned_hours),
                "actual_transit_hours": float(actual_hours),
                "base_cost": float(base_cost),
                "capacity_limit_units": profile["capacity_units"],
            })
            for rec in sub_chunk["record_id"]:
                rel_rows.append({"record_id": rec, "shipment_id": shipment_id})

    shipments = pd.DataFrame(rows)
    order_to_shipment = pd.DataFrame(rel_rows)
    return shipments, order_to_shipment


def make_trips(shipments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if shipments.empty:
        return pd.DataFrame(), pd.DataFrame()

    shipments = shipments.sort_values(["warehouse_id", "week_start", "requested_delivery_date"]).reset_index(drop=True)
    trip_rows: List[Dict] = []
    leg_rows: List[Dict] = []
    trip_counter = 0
    rng = np.random.default_rng(RANDOM_STATE)

    mode_map = {"bike": "road", "car": "road", "van": "road", "truck": "road"}

    for (warehouse_id, week_start), sub in shipments.groupby(["warehouse_id", "week_start"], dropna=False):
        sub = sub.reset_index(drop=True)
        for offset in range(0, len(sub), SHIPMENTS_PER_TRIP):
            chunk = sub.iloc[offset : offset + SHIPMENTS_PER_TRIP]
            if chunk.empty:
                continue
            trip_id = f"TRIP{trip_counter:07d}"
            trip_counter += 1
            total_units = int(chunk["total_units"].sum())
            if total_units <= VEHICLE_PROFILES["bike"]["capacity_units"]:
                vehicle = "bike"
            elif total_units <= VEHICLE_PROFILES["car"]["capacity_units"]:
                vehicle = "car"
            elif total_units <= VEHICLE_PROFILES["van"]["capacity_units"]:
                vehicle = "van"
            else:
                vehicle = "truck"

            profile = VEHICLE_PROFILES[vehicle]
            planned_departure = pd.Timestamp(week_start) + pd.Timedelta(hours=int(offset % 24) * 1.5)
            departure_jitter = rng.uniform(-0.5, 1.5)
            actual_departure = planned_departure + pd.Timedelta(hours=float(departure_jitter))

            total_distance = float(chunk["distance_km"].sum())
            total_cost = float(chunk["base_cost"].sum())
            sla_breach = int((chunk["on_time_flag"] == 0).sum())

            trip_rows.append({
                "trip_id": trip_id,
                "vehicle_type": vehicle,
                "planned_departure_time": planned_departure.isoformat(),
                "actual_departure_time": actual_departure.isoformat(),
                "warehouse_id": warehouse_id,
                "week_start": pd.Timestamp(week_start).isoformat() if pd.notna(week_start) else "",
                "total_shipments": int(len(chunk)),
                "total_units": total_units,
                "total_distance_km": total_distance,
                "total_cost": total_cost,
                "late_shipments": sla_breach,
                "driver_or_route_code": f"R-{hash(warehouse_id + str(week_start)) % 9999:04d}",
                "capacity_units": profile["capacity_units"],
            })

            leg_departure = planned_departure
            prev_actual_arrival = actual_departure
            for leg_index, shipment_row in enumerate(chunk.itertuples(index=False), start=1):
                planned_arrival = leg_departure + pd.Timedelta(hours=float(getattr(shipment_row, "planned_transit_hours", 0.0)))
                delay_hours = float(getattr(shipment_row, "delivery_delay_hours", 0.0))
                arrival_jitter = rng.uniform(-0.25, 1.0)
                planned_departure_leg = leg_departure
                actual_departure_leg = max(prev_actual_arrival, planned_departure_leg)
                actual_arrival = actual_departure_leg + pd.Timedelta(
                    hours=float(getattr(shipment_row, "planned_transit_hours", 0.0)) + max(delay_hours + arrival_jitter, -0.5)
                )
                prev_actual_arrival = actual_arrival
                co2_estimate = float(getattr(shipment_row, "distance_km", 0.0)) * (0.18 if vehicle in {"truck", "van"} else 0.05)

                leg_rows.append({
                    "shipment_id": shipment_row.shipment_id,
                    "trip_id": trip_id,
                    "leg_index": leg_index,
                    "origin_location_id": shipment_row.origin_location_id,
                    "destination_location_id": shipment_row.destination_location_id,
                    "mode": mode_map.get(vehicle, "road"),
                    "planned_departure": planned_departure_leg.isoformat(),
                    "planned_arrival": planned_arrival.isoformat(),
                    "actual_departure": actual_departure_leg.isoformat(),
                    "actual_arrival": actual_arrival.isoformat(),
                    "distance_km": float(getattr(shipment_row, "distance_km", 0.0)),
                    "planned_transit_hours": float(getattr(shipment_row, "planned_transit_hours", 0.0)),
                    "actual_transit_hours": float((actual_arrival - actual_departure_leg).total_seconds() / 3600.0),
                    "base_cost": float(getattr(shipment_row, "base_cost", 0.0)),
                    "capacity_limit_units": profile["capacity_units"],
                    "sla_minutes": float(getattr(shipment_row, "sla_minutes", 24 * 60)),
                    "leg_delay_hours": float((actual_arrival - planned_arrival).total_seconds() / 3600.0),
                    "on_time_flag": int((actual_arrival <= planned_arrival + pd.Timedelta(minutes=15))),
                    "co2_estimate_kg": co2_estimate,
                })

                leg_departure = planned_arrival

    trips = pd.DataFrame(trip_rows)
    shipment_to_trip = pd.DataFrame(leg_rows)
    return trips, shipment_to_trip


def make_batches(
    orders: pd.DataFrame,
    products: pd.DataFrame,
    suppliers: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prod_lookup = dict(zip(products["product_name"], products["product_id"]))
    supp_by_cat = suppliers["supplier_id"].tolist()
    rng = np.random.default_rng(RANDOM_STATE)

    weekly = (
        orders.assign(product_id=orders["product_name"].map(prod_lookup))
        .dropna(subset=["product_id"])
        .groupby(["product_id", "week_start"])["qty"]
        .sum()
        .reset_index()
    )

    batch_rows: List[Dict] = []
    plant_records: Dict[str, Dict] = {}
    batch_counter = 0

    for (pid, wk), sub in weekly.groupby(["product_id", "week_start"], dropna=False):
        demand_units = int(sub["qty"].sum())
        batch_size = max(BATCH_SIZE_UNITS, int(demand_units * 1.15))
        n_batches = max(1, math.ceil(max(demand_units, 1) / batch_size))
        for b in range(n_batches):
            batch_counter += 1
            batch_id = f"BATCH{batch_counter:07d}"
            plant_idx = (hash(pid) + b) % len(PLANT_LOCATIONS)
            plant_city, plant_country = PLANT_LOCATIONS[plant_idx]
            plant_id = f"PLANT{plant_idx+1:03d}"
            plant_loc_id = f"LOC_PLANT_{plant_idx+1:03d}"
            lat, lon = synth_xy(f"plant_{plant_id}")
            start = pd.Timestamp(wk) - pd.Timedelta(days=int(rng.integers(2, 6)))
            end = start + pd.Timedelta(days=int(rng.integers(1, 3)))
            actual_start = start + pd.Timedelta(hours=float(rng.uniform(-6, 12)))
            actual_end = end + pd.Timedelta(hours=float(rng.uniform(0, 18)))
            supplier_id = supp_by_cat[(hash(pid) + b) % len(supp_by_cat)]

            batch_rows.append({
                "batch_id": batch_id,
                "product_id": pid,
                "scheduled_start_time": start.isoformat(),
                "scheduled_end_time": end.isoformat(),
                "actual_start_time": actual_start.isoformat(),
                "actual_end_time": actual_end.isoformat(),
                "supplier_id": supplier_id,
                "plant_id": plant_id,
                "plant_location_id": plant_loc_id,
                "output_qty_units": int(batch_size),
                "status": "Completed" if actual_end <= end + pd.Timedelta(days=2) else "Delayed",
            })

            if plant_id not in plant_records:
                plant_records[plant_id] = {
                    "plant_id": plant_id,
                    "plant_name": f"Manufacturing Plant {plant_idx+1}",
                    "location_id": plant_loc_id,
                    "country": plant_country,
                    "city": plant_city,
                    "latitude": float(lat),
                    "longitude": float(lon),
                }

    batches = pd.DataFrame(batch_rows)
    plant_df = pd.DataFrame(list(plant_records.values())) if plant_records else pd.DataFrame(
        columns=["plant_id", "plant_name", "location_id", "country", "city", "latitude", "longitude"]
    )

    batch_to_shipment = pd.DataFrame(columns=["batch_id", "shipment_id"])
    return batches, batch_to_shipment, plant_df


def make_sla() -> pd.DataFrame:
    rows = []
    for name, minutes in SLA_OPTIONS.items():
        rows.append({
            "sla_name": name,
            "sla_minutes": int(minutes),
            "notes": f"Synthetic SLA target of {minutes/60:.1f} hours",
        })
    return pd.DataFrame(rows)


# --------------------------- Orchestration -----------------------------------

def main() -> None:
    print("Loading source orders...")
    orders = load_source_orders()
    n_orders = len(orders)
    print(f"Orders loaded: {n_orders}")

    print("Deriving customers and schedule windows...")
    customers, orders = make_customers(orders)
    orders = augment_orders_with_schedule(orders)
    print(f"Customers: {len(customers)}")

    print("Building products...")
    products = make_products(orders)
    print(f"Products: {len(products)}")

    print("Building suppliers...")
    suppliers = make_suppliers(products)
    print(f"Suppliers: {len(suppliers)}")

    print("Building warehouses...")
    warehouses = make_warehouses(orders)
    print(f"Warehouses: {len(warehouses)}")

    print("Building BOM...")
    bom = make_bom(products)
    print(f"BOM rows: {len(bom)}")

    print("Building shipments...")
    shipments, order_to_shipment = make_shipments(orders, warehouses, customers)
    print(f"Shipments: {len(shipments)}; order_to_shipment links: {len(order_to_shipment)}")

    print("Building trips...")
    trips, shipment_to_trip = make_trips(shipments)
    print(f"Trips: {len(trips)}; shipment_to_trip links: {len(shipment_to_trip)}")

    print("Building production batches...")
    batches, _, plant_df = make_batches(orders, products, suppliers)
    print(f"Batches: {len(batches)}")

    print("Building SLA table...")
    sla = make_sla()

    print("Building unified location dimension...")
    dim_locations = make_dim_locations(customers, suppliers, warehouses, plant_df)
    print(f"Locations: {len(dim_locations)}")

    # Build batch_to_shipment realistically:
    # if a shipment's week has orders of a given product, link the corresponding week's batches
    print("Linking batches to shipments...")
    prod_lookup = dict(zip(products["product_id"], products["product_name"]))

    # Map record → product_id
    product_id_by_order = orders["product_name"].map({r.product_name: r.product_id for _, r in products.iterrows()})
    orders_plus = pd.DataFrame({
        "record_id": orders["record_id"],
        "week_start": orders["week_start"],
        "product_id": product_id_by_order,
    }).dropna(subset=["product_id"])

    # Map order → shipment; then shipment → week + product presence
    order_to_shipment_plus = order_to_shipment.merge(orders_plus, on="record_id", how="left")
    # For each (shipment_id), get set of product_ids present
    ship_prod = (
        order_to_shipment_plus.dropna(subset=["shipment_id", "product_id"])
        .groupby(["shipment_id", "week_start"])["product_id"]
        .agg(lambda s: list(pd.Series(s).dropna().unique()))
        .reset_index()
        .rename(columns={"week_start": "ship_week"})
    )

    # For each (product_id, week) batches, link to shipments with same week & product present
    batches_plus = batches.copy()
    batches_plus["batch_week"] = pd.to_datetime(batches_plus["scheduled_end_time"]).dt.to_period("W").apply(lambda r: r.start_time)

    batch_to_shipment_rows = []
    for _, br in batches_plus.iterrows():
        pid = br["product_id"]
        wk = br["batch_week"]
        # find shipments in same week that include the product
        try:
            # This is more efficient with a join, but we'll do a safe scan for clarity
            mask = (ship_prod["ship_week"] == wk) & ship_prod["product_id"].apply(lambda L: pid in L)
            hits = ship_prod.loc[mask, "shipment_id"].tolist()
        except Exception:
            hits = []
        # link to a reasonable subset (avoid exploding links)
        if hits:
            take = hits[: max(1, len(hits)//5) ]  # cap many-to-many
            for sh in take:
                batch_to_shipment_rows.append({"batch_id": br["batch_id"], "shipment_id": sh})

    batch_to_shipment = pd.DataFrame(batch_to_shipment_rows).drop_duplicates()
    print(f"batch_to_shipment links: {len(batch_to_shipment)}")

    # -------------------- Write all CSVs --------------------
    print("Writing CSV files to:", OUT_DIR)

    files = {
        "suppliers.csv": suppliers,
        "warehouses.csv": warehouses,
        "products.csv": products,
        "bom.csv": bom,
        "production_batches.csv": batches,
        "trips.csv": trips,
        "sla.csv": sla,
        "order_to_shipment.csv": order_to_shipment,
        "shipment_to_trip.csv": shipment_to_trip,
        "batch_to_shipment.csv": batch_to_shipment,
        "customers.csv": customers,
        "dim_locations.csv": dim_locations,
        "plants.csv": plant_df,
        "shipments.csv": shipments,
        "orders_enriched.csv": orders,
    }
    manifest = {}
    for name, df in files.items():
        manifest[name] = {
            "rows": int(len(df)),
            "path": write_csv(df, name),
        }

    # Optional: write a manifest JSON for quick inspection
    with open(os.path.join(OUT_DIR, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    for k, v in manifest.items():
        print(f"{k:28s} -> rows={v['rows']:6d}  path={v['path']}")


if __name__ == "__main__":
    main()
