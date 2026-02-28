---
name: Stabilize Forecast Generalization
overview: Make demand model performance on new bookings more consistent by replacing fragile single-split evaluation with walk-forward validation, adding drift checks, and retraining guardrails. Focus is all time-series targets with manual retrain flow.
todos:
  - id: wf-validation
    content: Implement expanding walk-forward validation in train_ts_models and store fold metrics per target
    status: completed
  - id: regularize-prune
    content: Add validation-driven regularization and prune unstable low-value lag features
    status: completed
  - id: drift-alerts
    content: Add drift checks on new uploaded booking data and show warnings in UI
    status: completed
  - id: model-promotion
    content: Add manual retrain promotion gate comparing old vs new robust metrics
    status: completed
  - id: ui-diagnostics
    content: Update diagnostics panel to show fold distributions and robust score summaries
    status: completed
  - id: docs-update
    content: Update README with OOS evaluation method and non-guarantee statement for fixed R2
    status: completed
isProject: false
---

# Stabilize Out-of-Sample R2 for New Booking Data

## Goal

Raise and stabilize generalization quality on new booking data for all time-series targets, while acknowledging there is no mathematically guaranteed fixed `R2 >= 0.80` under regime shifts.

## Why current setup drops on new data

- Evaluation in `[hotel_demand_forecast.py](hotel_demand_forecast.py)` uses one fixed tail split (`split = len(df) - 90`) inside `train_ts_models`, which can overfit to that single period.
- Feature generation is very high-dimensional (`_add_lags` creates many lag/rolling/diff features), increasing overfitting risk when patterns shift.
- No drift detection/alerting exists before using models on new booking distributions.

## Implementation plan

- Update `train_ts_models` in `[hotel_demand_forecast.py](hotel_demand_forecast.py)` to use walk-forward validation (expanding window) and compute fold-level metrics (`R2`, `MAE`, `MAPE`) per target.
- Keep one final holdout block for reporting, but surface both holdout and walk-forward stats in UI so tuning is based on robust OOS metrics, not one lucky split.
- Add target-wise regularization controls (fewer trees / stronger `min_child_samples`, `reg_alpha`, `reg_lambda`, optional feature subsampling) driven by validation score variance.
- Add feature-pruning step using feature importance stability across folds (drop unstable low-value lag features).
- Add drift checks between training window vs newest uploaded data (PSI-like/mean-std shift + calendar mix shifts) and show warning badge when drift is high.
- Add manual retrain gating in sidebar: only replace current models if walk-forward median `R2` improves or stays within a tolerance while error metrics improve.
- Extend diagnostics tab to show per-target fold chart and “expected range” instead of single point estimate, improving consistency expectations.

## Acceptance criteria

- Diagnostics shows fold-wise out-of-sample metrics for all time-series targets.
- Retrain action reports: previous vs new model metrics and whether promotion occurred.
- Drift panel appears when new booking distribution differs materially from training baseline.
- Reported model quality for each target is based on robust OOS validation (not single split only).

## Files to modify

- `[hotel_demand_forecast.py](hotel_demand_forecast.py)` (primary: training/evaluation, diagnostics UI, retrain promotion logic)
- `[README.md](README.md)` (brief explanation of validation methodology and realistic performance expectations)

