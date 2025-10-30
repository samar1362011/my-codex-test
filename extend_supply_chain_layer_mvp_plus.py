# extend_supply_chain_layer_mvp_plus.py
# -----------------------------------------------------------------------------
# Adds five "MVP++" tables to the independent supply-chain layer, without
# modifying your original dataset or the existing generated CSVs.
#
# New tables generated under RES/supply_chain_layer/:
#   - purchase_orders.csv
#   - po_to_batch.csv
#   - work_orders.csv
#   - inventory_movements.csv
#   - resource_calendar.csv
#
# Requirements:
#   - You have already run build_supply_chain_layer.py and have the following:
#     products.csv, suppliers.csv, warehouses.csv, production_batches.csv,
#     batch_to_shipment.csv, shipment_to_trip.csv, trips.csv
#
# Run:
#   python extend_supply_chain_layer_mvp_plus.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ------------------------------- Config --------------------------------------

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.getcwd()
LAYER_DIR = os.path.join(BASE_DIR, "RES", "supply_chain_layer")
os.makedirs(LAYER_DIR, exist_ok=True)

# Tunables
PO_BATCH_RATIO = 0.33          # ~1 PO per 3 batches (per product), min 1
WO_JITTER_DAYS = (0, 2)        # actual_start = scheduled_start +/- jitter
INV_OUT_LAG_HOURS = (0, 6)     # OUT movement occurs close to trip departure
CALENDAR_WEEKS = 3             # generate simple weekly calendar templates


# ---------------------------- IO Utilities -----------------------------------

def _read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(LAYER_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required CSV not found: {path}")
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"Failed to read {path} with common encodings.")


def _write_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(LAYER_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def _to_ts(s) -> pd.Timestamp:
    return pd.to_datetime(s, errors="coerce")


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with whitespace-trimmed column labels."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _ensure_column(df: pd.DataFrame, target: str, default, *aliases: str) -> None:
    """Ensure ``target`` exists, copying from the first alias that is present."""
    if target in df.columns:
        return
    for name in aliases:
        if name in df.columns:
            df[target] = df[name]
            return
    df[target] = default


def _ensure_datetime_column(df: pd.DataFrame, target: str, *aliases: str) -> None:
    """Populate ``target`` with datetime values, accepting legacy alias columns."""
    _ensure_column(df, target, pd.NaT, *aliases)
    df[target] = _to_ts(df[target])


def _ensure_numeric_column(df: pd.DataFrame, target: str, *aliases: str) -> None:
    """Populate ``target`` with numeric values, accepting legacy alias columns."""
    _ensure_column(df, target, np.nan, *aliases)
    df[target] = pd.to_numeric(df[target], errors="coerce")


# --------------------------- Generators --------------------------------------

def build_purchase_orders(
    products: pd.DataFrame,
    suppliers: pd.DataFrame,
    batches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create purchase_orders and po_to_batch mapping."""
    if batches.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Supplier selection: simple round-robin
    supp_ids = suppliers["supplier_id"].astype(str).tolist()
    if not supp_ids:
        # fallback single supplier
        supp_ids = ["S1000"]

    batches = _strip_columns(batches).copy()
    # Normalise legacy column names
    legacy_map = {
        "production_batch_id": "batch_id",
        "scheduled_start": "scheduled_start_time",
        "scheduled_end": "scheduled_end_time",
        "planned_start_time": "scheduled_start_time",
        "planned_end_time": "scheduled_end_time",
        "actual_start": "actual_start_time",
        "actual_end": "actual_end_time",
        "output_units": "output_qty_units",
        "planned_output_units": "output_qty_units",
    }
    for old, new in legacy_map.items():
        if old in batches.columns and new not in batches.columns:
            batches[new] = batches[old]

    _ensure_column(
        batches,
        "batch_id",
        [f"LEGACY_BATCH_{i:07d}" for i in range(len(batches))],
        "production_batch_id",
    )
    batches["product_id"] = batches["product_id"].astype(str)
    _ensure_datetime_column(
        batches,
        "scheduled_start_time",
        "scheduled_start",
        "planned_start_time",
        "scheduled_start_ts",
    )
    _ensure_datetime_column(
        batches,
        "scheduled_end_time",
        "scheduled_end",
        "planned_end_time",
        "scheduled_end_ts",
    )
    _ensure_numeric_column(batches, "output_qty_units", "output_qty_units", "planned_output_units", "output_units")
    batches["output_qty_units"] = batches["output_qty_units"].fillna(0).astype(int)

    rng = np.random.default_rng(RANDOM_STATE)
    po_rows: List[Dict] = []
    link_rows: List[Dict] = []

    for pid, grp in batches.groupby("product_id", dropna=False):
        grp = grp.sort_values("scheduled_start_time")
        n_batches = len(grp)
        n_pos = max(1, math.ceil(n_batches * PO_BATCH_RATIO))

        # Split batches into n_pos buckets in order
        buckets = np.array_split(grp.reset_index(drop=True), n_pos)

        for i, bucket in enumerate(buckets):
            po_id = f"PO{hash(pid + str(i)) & 0x7FFFFFFF:09d}"
            supplier_id = supp_ids[(hash(pid) + i) % len(supp_ids)]

            # Order date slightly before first batch starts
            first_start = pd.to_datetime(bucket["scheduled_start_time"]).min()
            if pd.isna(first_start):
                first_start = pd.Timestamp("2016-01-01")
            lead_days = int(2 + (hash(po_id) % 7))  # synthetic lead time if not in suppliers
            order_date = first_start - pd.Timedelta(days=lead_days + 2)
            promised_date = order_date + pd.Timedelta(days=lead_days)

            ordered_qty_units = int(bucket["output_qty_units"].sum() * 1.05)  # small buffer
            unit_cost = float(10 + (hash(pid) % 90))  # synthetic unit cost

            po_rows.append({
                "po_id": po_id,
                "supplier_id": supplier_id,
                "product_id": pid,
                "order_date": order_date.isoformat(),
                "promised_date": promised_date.isoformat(),
                "ordered_qty_units": ordered_qty_units,
                "unit_cost": unit_cost,
            })

            # Link each batch in this bucket to the PO
            for _, br in bucket.iterrows():
                alloc = int(min(ordered_qty_units, int(br["output_qty_units"])))
                link_rows.append({
                    "po_id": po_id,
                    "batch_id": br["batch_id"],
                    "allocated_qty_units": alloc,
                })

    purchase_orders = pd.DataFrame(po_rows)
    po_to_batch = pd.DataFrame(link_rows)
    return purchase_orders, po_to_batch


def build_work_orders(batches: pd.DataFrame) -> pd.DataFrame:
    """Create work_orders for each production batch (1:1)."""
    if batches.empty:
        return pd.DataFrame()

    batches = _strip_columns(batches).copy()
    _ensure_column(
        batches,
        "batch_id",
        [f"LEGACY_BATCH_{i:07d}" for i in range(len(batches))],
        "production_batch_id",
    )
    batches["batch_id"] = batches["batch_id"].astype(str)
    _ensure_datetime_column(
        batches,
        "scheduled_start_time",
        "scheduled_start",
        "planned_start_time",
        "scheduled_start_ts",
    )
    _ensure_datetime_column(
        batches,
        "scheduled_end_time",
        "scheduled_end",
        "planned_end_time",
        "scheduled_end_ts",
    )
    _ensure_datetime_column(
        batches,
        "actual_start_time",
        "actual_start",
        "actual_begin_time",
    )
    _ensure_datetime_column(
        batches,
        "actual_end_time",
        "actual_end",
        "actual_finish_time",
    )
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for _, r in batches.iterrows():
        sched_start = _to_ts(r.get("scheduled_start_time"))
        sched_end = _to_ts(r.get("scheduled_end_time"))
        if pd.isna(sched_start) or pd.isna(sched_end):
            continue
        # Actuals with small jitter
        jstart = rng.integers(WO_JITTER_DAYS[0], WO_JITTER_DAYS[1] + 1)
        jend = rng.integers(WO_JITTER_DAYS[0], WO_JITTER_DAYS[1] + 1)
        base_actual_start = _to_ts(r.get("actual_start_time"))
        actual_start = (
            base_actual_start
            if pd.notna(base_actual_start)
            else sched_start + pd.Timedelta(days=int(jstart))
        )
        # keep duration close to scheduled
        duration = max((sched_end - sched_start).days, 1)
        base_actual_end = _to_ts(r.get("actual_end_time"))
        if pd.notna(base_actual_end):
            actual_end = base_actual_end
        else:
            actual_end = actual_start + pd.Timedelta(days=int(max(duration + jend, 1)))
        status = "Completed" if actual_end <= (sched_end + pd.Timedelta(days=3)) else "Delayed"

        rows.append({
            "wo_id": f"WO{hash(str(r['batch_id'])) & 0x7FFFFFFF:09d}",
            "batch_id": r["batch_id"],
            "scheduled_start": sched_start.isoformat(),
            "scheduled_end": sched_end.isoformat(),
            "actual_start": actual_start.isoformat(),
            "actual_end": actual_end.isoformat(),
            "status": status,
        })

    return pd.DataFrame(rows)


def make_dim_locations(
    customers: pd.DataFrame,
    suppliers: pd.DataFrame,
    warehouses: pd.DataFrame,
    plant_entries: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    if customers is not None and not customers.empty:
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

    if suppliers is not None and not suppliers.empty:
        frames.append(
            suppliers[
                [
                    "location_id",
                    "supplier_id",
                    "supplier_name",
                    "region",
                    "country",
                    "city",
                    "latitude",
                    "longitude",
                ]
            ]
            .rename(
                columns={
                    "supplier_id": "entity_id",
                    "supplier_name": "entity_name",
                }
            )
            .assign(entity_type="supplier")
        )

    if warehouses is not None and not warehouses.empty:
        frames.append(
            warehouses[
                [
                    "location_id",
                    "warehouse_id",
                    "warehouse_name",
                    "region",
                    "country",
                    "city",
                    "latitude",
                    "longitude",
                ]
            ]
            .rename(
                columns={
                    "warehouse_id": "entity_id",
                    "warehouse_name": "entity_name",
                }
            )
            .assign(entity_type="warehouse")
        )

    if plant_entries is not None and not plant_entries.empty:
        frames.append(
            plant_entries[
                [
                    "location_id",
                    "plant_id",
                    "plant_name",
                    "country",
                    "state",
                    "city",
                    "latitude",
                    "longitude",
                ]
            ]
            .rename(
                columns={
                    "plant_id": "entity_id",
                    "plant_name": "entity_name",
                }
            )
            .assign(entity_type="plant")
        )

    if not frames:
        return pd.DataFrame(
            columns=
            [
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
    return dim_locations.drop_duplicates(subset=["location_id"])  # ensure stability


def build_inventory_movements(
    batches: pd.DataFrame,
    batch_to_shipment: pd.DataFrame,
    shipment_to_trip: pd.DataFrame,
    trips: pd.DataFrame,
) -> pd.DataFrame:
    """Create IN/OUT movements linking production to dispatch trips."""
    if batches.empty:
        return pd.DataFrame()

    batches = _strip_columns(batches).copy()
    batch_to_shipment = _strip_columns(batch_to_shipment)
    _ensure_column(
        batches,
        "batch_id",
        [f"LEGACY_BATCH_{i:07d}" for i in range(len(batches))],
        "production_batch_id",
    )
    _ensure_column(
        batch_to_shipment,
        "batch_id",
        [np.nan] * len(batch_to_shipment),
        "production_batch_id",
    )
    batches["batch_id"] = batches["batch_id"].astype(str)
    batch_to_shipment["batch_id"] = (
        batch_to_shipment["batch_id"].astype(str).where(~batch_to_shipment["batch_id"].isna())
    )
    _ensure_datetime_column(
        batches,
        "scheduled_end_time",
        "scheduled_end",
        "planned_end_time",
        "scheduled_end_ts",
    )
    _ensure_datetime_column(batches, "actual_end_time", "actual_end", "actual_finish_time")
    _ensure_numeric_column(batches, "output_qty_units", "output_qty_units", "planned_output_units", "output_units")
    batches["output_qty_units"] = batches["output_qty_units"].fillna(0).astype(int)

    # Join shipments -> trips to get warehouse & departure_time
    st = shipment_to_trip.copy()
    trips = trips.copy()
    _ensure_datetime_column(
        trips,
        "actual_departure_time",
        "departure_time",
        "planned_departure_time",
        "scheduled_departure_time",
    )

    st = st.merge(trips[["trip_id", "warehouse_id", "actual_departure_time"]], on="trip_id", how="left")

    # Link batches to shipments, then to trips
    b2s = batch_to_shipment.merge(st, on="shipment_id", how="left")

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    # IN movement: at batch completion into a pseudo storage (choose a warehouse from the linked trips if available)
    # OUT movement: when shipments assigned to a trip depart
    for bid, g in b2s.groupby("batch_id", dropna=False):
        b_row = batches[batches["batch_id"] == bid]
        if b_row.empty:
            continue
        b = b_row.iloc[0]
        total_out = 0

        # If we have linked trips, create OUT movements per shipment/trip
        for _, link in g.iterrows():
            wh = str(link.get("warehouse_id", "W100"))
            dep = _to_ts(link.get("actual_departure_time"))
            if pd.isna(dep):
                # fallback if trip time is missing
                dep = _to_ts(b.get("actual_end_time"))
                if pd.isna(dep):
                    dep = _to_ts(b.get("scheduled_end_time")) + pd.Timedelta(days=1)

            # Split some quantity (synthetic) per linked shipment
            remaining_capacity = max(int(b["output_qty_units"]) - total_out, 0)
            if remaining_capacity <= 0:
                break
            qty = max(
                int(remaining_capacity * (0.6 + 0.4 * rng.random()) / max(len(g), 1)),
                1,
            )
            qty = min(qty, remaining_capacity)
            total_out += qty

            lag_hours = int(rng.integers(INV_OUT_LAG_HOURS[0], INV_OUT_LAG_HOURS[1] + 1))
            rows.append({
                "move_id": f"MOV_OUT_{hash(bid + str(link.get('shipment_id')))&0x7FFFFFFF:09d}",
                "warehouse_id": wh,
                "product_id": b["product_id"],
                "movement_type": "OUT",
                "quantity": qty,
                "move_time": (dep + pd.Timedelta(hours=lag_hours)).isoformat(),
                "reference": f"shipment:{link.get('shipment_id')}|trip:{link.get('trip_id')}",
            })

        # IN movement for the remainder (or full qty if no links)
        in_qty = max(int(b["output_qty_units"] - total_out), 0)
        wh_series = g.get("warehouse_id", pd.Series(dtype=object))
        if not isinstance(wh_series, pd.Series):
            wh_series = pd.Series(dtype=object)
        wh_non_null = wh_series.dropna()
        wh_in = str(wh_non_null.iloc[0]) if not wh_non_null.empty else "W100"
        completion_time = _to_ts(b.get("actual_end_time"))
        if pd.isna(completion_time):
            completion_time = _to_ts(b.get("scheduled_end_time"))
        rows.append({
            "move_id": f"MOV_IN_{hash(bid)&0x7FFFFFFF:09d}",
            "warehouse_id": wh_in,
            "product_id": b["product_id"],
            "movement_type": "IN",
            "quantity": int(in_qty if in_qty > 0 else int(b["output_qty_units"])),
            "move_time": completion_time.isoformat() if pd.notna(completion_time) else pd.Timestamp.utcnow().isoformat(),
            "reference": f"batch:{bid}",
        })

    return pd.DataFrame(rows)


def build_resource_calendar(
    suppliers: pd.DataFrame,
    warehouses: pd.DataFrame,
    trips: pd.DataFrame,
) -> pd.DataFrame:
    """Simple weekly calendars for suppliers/warehouses/vehicle types."""
    rows = []

    # Suppliers: Mon–Fri 08:00–18:00, capacity from suppliers
    for _, s in suppliers.iterrows():
        cap = int(pd.to_numeric(s.get("max_daily_capacity_units", 2000), errors="coerce"))
        for w in range(CALENDAR_WEEKS):
            for d in range(5):  # Mon-Fri
                rows.append({
                    "resource_type": "supplier",
                    "resource_id": s["supplier_id"],
                    "working_day": (pd.Timestamp.today().normalize() + pd.Timedelta(weeks=w, days=d)).date().isoformat(),
                    "open_time": "08:00",
                    "close_time": "18:00",
                    "capacity_units": cap,
                })

    # Warehouses: Mon–Sat 07:00–22:00
    for _, w in warehouses.iterrows():
        cap = int(pd.to_numeric(w.get("capacity_units", 5000), errors="coerce"))
        for wk in range(CALENDAR_WEEKS):
            for d in range(6):  # Mon-Sat
                rows.append({
                    "resource_type": "warehouse",
                    "resource_id": w["warehouse_id"],
                    "working_day": (pd.Timestamp.today().normalize() + pd.Timedelta(weeks=wk, days=d)).date().isoformat(),
                    "open_time": "07:00",
                    "close_time": "22:00",
                    "capacity_units": cap,
                })

    # Vehicles: unique vehicle types from trips, generic capacity windows
    vehicles = sorted(set(trips["vehicle_type"].astype(str).tolist()))
    for vt in vehicles or ["van", "car", "bike"]:
        for wk in range(CALENDAR_WEEKS):
            for d in range(7):  # daily
                rows.append({
                    "resource_type": "vehicle",
                    "resource_id": vt,
                    "working_day": (pd.Timestamp.today().normalize() + pd.Timedelta(weeks=wk, days=d)).date().isoformat(),
                    "open_time": "06:00",
                    "close_time": "23:00",
                    "capacity_units": 999999,  # interpreted as availability rather than per-vehicle capacity
                })

    return pd.DataFrame(rows)


# --------------------------- Orchestration -----------------------------------

def main() -> None:
    print("Loading existing layer CSVs from:", LAYER_DIR)
    products = _strip_columns(_read_csv("products.csv"))
    suppliers = _strip_columns(_read_csv("suppliers.csv"))
    warehouses = _strip_columns(_read_csv("warehouses.csv"))
    batches = _strip_columns(_read_csv("production_batches.csv"))
    batch_to_shipment = _strip_columns(_read_csv("batch_to_shipment.csv"))
    shipment_to_trip = _strip_columns(_read_csv("shipment_to_trip.csv"))
    trips = _strip_columns(_read_csv("trips.csv"))

    _ensure_column(
        batches,
        "batch_id",
        [f"LEGACY_BATCH_{i:07d}" for i in range(len(batches))],
        "production_batch_id",
    )
    _ensure_column(
        batch_to_shipment,
        "batch_id",
        [np.nan] * len(batch_to_shipment),
        "production_batch_id",
    )

    print("Building purchase_orders and po_to_batch...")
    purchase_orders, po_to_batch = build_purchase_orders(products, suppliers, batches)
    print(f"purchase_orders: {len(purchase_orders)} rows  |  po_to_batch: {len(po_to_batch)} rows")

    print("Building work_orders...")
    work_orders = build_work_orders(batches)
    print(f"work_orders: {len(work_orders)} rows")

    print("Building inventory_movements...")
    inventory_movements = build_inventory_movements(batches, batch_to_shipment, shipment_to_trip, trips)
    print(f"inventory_movements: {len(inventory_movements)} rows")

    print("Building resource_calendar...")
    resource_calendar = build_resource_calendar(suppliers, warehouses, trips)
    print(f"resource_calendar: {len(resource_calendar)} rows")

    print("Writing CSVs...")
    out_files = {
        "purchase_orders.csv": purchase_orders,
        "po_to_batch.csv": po_to_batch,
        "work_orders.csv": work_orders,
        "inventory_movements.csv": inventory_movements,
        "resource_calendar.csv": resource_calendar,
    }
    for name, df in out_files.items():
        _write_csv(df, name)
        print(f"  -> {name}: {len(df)} rows")

    print("Done.")


if __name__ == "__main__":
    main()
