# Supply Chain Federated Learning Demo

This repository contains an end-to-end pipeline for preparing a supply-chain dataset,
performing routing optimization, and orchestrating a simple Flower-based federated
learning experiment.

## Execution order
Follow these steps sequentially to reproduce the full workflow:

1. **Run the pipeline**
   ```bash
   python pipeline.py
   ```
   * Creates cleaned samples under `RES/preprocessing/`.
   * Builds client feature tables inside `RES/clients/` and exports the global
     feature list plus transformation statistics under `RES/`.
   * Generates transformed datasets in `RES/transformation/`.
   * Creates the integrated `RES/integration/orders_journey.csv` view (when the
     supply-chain layer is available) and precomputes routing input drafts under
     `RES/optimization/inputs/` for both trip-based and warehouse-week grouping
     strategies.
   * Produces routing optimization outputs (summary, per-batch paths, and metrics)
     in `RES/optimization/routes/`.

2. **(Optional) Generate the synthetic supply-chain layer**
   ```bash
   python build_supply_chain_layer.py
   python extend_supply_chain_layer_mvp_plus.py
   ```
   These scripts derive additional supplier/warehouse logistics tables under
   `RES/supply_chain_layer/`, including the extended MVP++ entities such as
   purchase orders, work orders, inventory movements, and resource calendars.

### Understanding the synthetic supply-chain layer

Treat every CSV under `RES/supply_chain_layer/` as an individual database table.
The base orders (cleaned during preprocessing) remain the authoritative source
for order-level facts such as `record_id`, `order_date`, `ship_date`, `qty`, and
location fields. The synthetic layer builds relationships around those orders
without overwriting the original data. Key links include:

| From table | Join key(s) | To table | Purpose |
|------------|-------------|----------|---------|
| `orders` (from `RES/preprocessing/*.csv`) | `record_id` | `order_to_shipment` | Attach each order to its shipment group |
| `order_to_shipment` | `shipment_id` | `shipments` | Inspect shipment destinations, weeks, and aggregate load |
| `shipments` | `shipment_id` | `shipment_to_trip` → `trips` | Reveal the transport trip, vehicle, and departure time |
| `orders` | `product_name` / `product_id` | `products` | Retrieve product definitions and weights |
| `products` | `product_id` | `production_batches` | Trace manufacturing batches that produce the ordered items |
| `production_batches` | `batch_id` | `batch_to_shipment` | Map batches to shipments they fulfil |
| `production_batches` | `supplier_id` / `plant_id` | `suppliers` / `plants` / `purchase_orders` / `po_to_batch` | Follow sourcing back to suppliers, plants, and purchase orders |
| `production_batches` | `batch_id` | `work_orders` | Compare scheduled vs. actual execution |
| `production_batches` & `batch_to_shipment` | `batch_id`, `shipment_id` | `inventory_movements` | Audit IN/OUT stock movements at warehouses |
| `dim_locations` | `location_id` | `suppliers` / `warehouses` / `customers` / `plants` / `shipment_to_trip` | Normalise geography for leg-by-leg tracking |

Because many of these relationships are one-to-many, create focused analytical
views on demand instead of flattening everything into a single wide table. For
example:

* **End-to-end order journey:** join orders → order_to_shipment → shipments →
  shipment_to_trip → trips to see where and when the goods were dispatched.
* **Supplier traceability:** orders → order_to_shipment → batch_to_shipment →
  production_batches → suppliers (and optionally purchase_orders/po_to_batch)
  to uncover which supplier served each customer order.
* **Inventory tracking:** production_batches → batch_to_shipment →
  inventory_movements to understand when goods were stocked and shipped from
  each warehouse.

* **Customer segmentation:** use `customers` (tier, segment, geography) →
  `dim_locations` → `shipments` to analyse service levels by client cohort or
  market while preserving spatial context.

When modelling, align any engineered features from the transformation outputs
(`RES/transformation/`) via `record_id` after constructing the specific view you
need. This keeps the data modular, avoids exploding row counts, and preserves
the natural hierarchy of the supply chain layer.

Whenever the synthetic layer is present, `pipeline.py` automatically materialises
an order-journey view (`RES/integration/orders_journey.csv`) and two routing input
helpers in `RES/optimization/inputs/`: one grouped by existing `trip_id`
assignments and one grouped by `(warehouse, week_start)`. These files act as
ready-to-use bridges between preprocessing, transformation, and optimization.

3. **Inspect optimization artifacts (optional but recommended)**
   * `RES/optimization/routes/routes_summary.csv` — high-level KPIs for every batch.
   * `RES/optimization/routes/<group>/batch_XXX_path.csv` — step-by-step coordinates
     for each optimized delivery batch.
   * `RES/optimization/routes/<group>/batch_XXX_metrics.csv` — vehicle-level metrics
     justifying the selected route.

   To generate presentation-ready figures (route maps, time–cost trade-offs, SLA
   compliance), run:
   ```bash
   python plot_optimization_graphs.py
   ```
   The resulting images are written to `RES/optimization/routes/plots/`.

4. **Start the Flower server** (from a new terminal in the repo root)
   ```bash
   python server.py
   ```
   The server loads the global feature configuration, initializes the regression
   model, and listens for client updates.

5. **Launch clients** (one terminal per client)
   ```bash
   python client1.py
   python client2.py
   ```
   Each client reads its transformed dataset, trains locally, and participates in
   the federated rounds coordinated by the server.

6. **Review training plots** (optional)
   * Server metrics are saved under `RES/plots/server_*.png`.
   * Client-specific training curves are saved under `RES/plots/client_*/*.png`.

## Requirements
* Python 3.9+
* The `requirements.txt` from the course environment (TensorFlow, Flower,
  scikit-learn, pandas, numpy, matplotlib).

Ensure the `DATASET/DS_DataCoSupplyChainDataset.csv` file is available before
running the pipeline.
