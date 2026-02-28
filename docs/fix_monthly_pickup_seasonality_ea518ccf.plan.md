---
name: Fix monthly pickup seasonality
overview: Align pickup feature engineering across training and inference so monthly seasonality is preserved and forecasts no longer look flat across future months.
todos:
  - id: locate-shared-fillrate-logic
    content: Create a reusable period-aware pace baseline helper in hotel_demand_forecast.py
    status: completed
  - id: update-training-fillrate
    content: Switch build_pickup_training_data() fill_rate from global pace to month-aware pace with fallback
    status: completed
  - id: update-inference-fillrate
    content: Switch forecast_with_pickup_models() fill_rate to the exact same month-aware logic
    status: completed
  - id: validate-monthly-separation
    content: Re-train and verify that monthly occupancy/revenue pickup forecasts are differentiated
    status: completed
isProject: false
---

# Fix monthly pickup seasonality

## What is happening now

In both training and inference, `fill_rate` is currently derived from a single global `avg_daily_pace`, so the expected OTB baseline does not vary by season/month. This makes month-level behavior too similar.

- Training code path: `[D:/Python/Demand-forecast/hotel_demand_forecast.py](D:/Python/Demand-forecast/hotel_demand_forecast.py)` in `build_pickup_training_data()` computes one `avg_daily_pace` then applies it to every `(stay_date, dta)` row.
- Inference code path: `[D:/Python/Demand-forecast/hotel_demand_forecast.py](D:/Python/Demand-forecast/hotel_demand_forecast.py)` in `forecast_with_pickup_models()` mirrors the same global scalar, so the feature remains non-seasonal.

## Implementation plan

1. Add a shared helper in `[D:/Python/Demand-forecast/hotel_demand_forecast.py](D:/Python/Demand-forecast/hotel_demand_forecast.py)` to compute pace baselines by period (default month), with fallback to global pace when a period is sparse/missing.
2. Update `build_pickup_training_data()` to compute `fill_rate` using `avg_daily_pace` selected by `stay_date.month` (fallback to global).
3. Update `forecast_with_pickup_models()` to use the exact same helper and lookup logic, ensuring train/inference feature parity.
4. Keep feature schema unchanged (`PICKUP_FEATURES` still includes `fill_rate`) so model interfaces and downstream UI stay compatible.
5. Add lightweight validation checks (debug/log or assertions) to confirm `fill_rate` distribution varies by month for both generated training rows and forecast rows.
6. Retrain pickup models and compare monthly pickup/revenue outputs to verify month separation reappears.

## Validation steps

- Run a forecast spanning at least 3-4 months and confirm `pickup_occ_pct` / `pickup_revenue` monthly aggregates are no longer nearly identical.
- Compare feature summaries by month (`fill_rate`, `pace_7d_pct`, `otb_occ_pct`) before vs after change.
- Sanity-check sparse-month behavior uses fallback baseline and does not produce divide-by-zero artifacts.

