# Hotel Revenue Intelligence Platform

A Streamlit app for hotel demand forecasting and revenue intelligence. Train LightGBM models on your bookings data to forecast revenue, occupancy, ADR, channel mix, and to score cancellation risk and length-of-stay for individual bookings.

## Features

- **Time-series models (stay-date level)**  
  Nightly revenue, overall and room-type occupancy %, ADR, and channel mix (Direct, OTA, Walk-in, Website) — all forecast with LightGBM regressors.

- **Booking-level models**  
  Cancellation probability (LightGBM classifier) and length-of-stay prediction (LightGBM regressor).

- **Robust out-of-sample evaluation**  
  Walk-forward cross-validation with expanding windows, feature pruning based on importance stability, and drift detection for incoming booking data.

- **Model promotion gate**  
  New models are only promoted if they meet quality criteria (stable R² across folds, no significant degradation in error metrics).

- **Interactive dashboard**  
  Tabs for demand forecast, ADR forecast, cancellation risk, length-of-stay, channel mix, diagnostics with fold-level metrics, feature importance, and data explorer.

- **Single-booking scoring**  
  Enter check-in/out, channel, rate plan, and get cancellation probability and predicted LOS with risk tier and recommendations.

## Important: No Fixed R² Guarantee

There is **no mathematically guaranteed fixed R²** (e.g., "always ≥ 0.80") when applying trained models to new booking data. Forecast accuracy depends on:

- **Data distribution similarity**: If new data has different patterns (seasonality, booking behavior, pricing), accuracy will change.
- **Temporal stability**: Hotel demand patterns shift due to holidays, events, economic conditions, and competition.
- **Regime changes**: Major disruptions (new competitors, renovation, pandemics) can make historical patterns irrelevant.

This platform uses **robust evaluation methods** (walk-forward CV, drift detection) to give you realistic expectations about model performance on new data, but cannot guarantee a fixed R² threshold under all conditions.

## Requirements

- Python 3.9+
- Dependencies in `requirements.txt`: Streamlit, pandas, numpy, LightGBM, scikit-learn, Altair

## Installation

```bash
# Clone or navigate to the project
cd Demand-forecast

# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run hotel_demand_forecast.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Data format

Upload a **Bookings CSV** with at least these columns (names must match):

| Column               | Description                          |
|----------------------|--------------------------------------|
| `Booking_ID`         | Unique booking identifier            |
| `Check_in_Date`      | Check-in date                        |
| `Check_out_Date`     | Check-out date                       |
| `Booking_Date`       | Date of booking                      |
| `Room_Type`          | e.g. Standard, Deluxe, Suite          |
| `Booking_Channel`    | e.g. Direct, OTA, Walk-in, Website    |
| `Rate_Plan`          | e.g. BAR, Corporate, Early Bird, …   |
| `Booked_Rate`        | Booked rate (numeric, e.g. IDR)      |
| `Revenue_Generated`  | Revenue (numeric)                    |
| `Number_of_Nights`   | Length of stay                       |
| `Number_of_Guests`   | Guest count                          |
| `Cancellation_Status`| e.g. Confirmed, Cancelled            |

Dates should be parseable by pandas (e.g. `YYYY-MM-DD`).

## Workflow

1. **Upload** your Bookings CSV in the sidebar.
2. **Review drift warnings** — if high data drift is detected, the system will warn you.
3. **Train** — click **Train All Models** to build the 12 LightGBM models with walk-forward validation.
4. **Check promotion status** — new models are only promoted if they pass quality gates (stable robust R², no significant MAPE degradation).
5. **Explore** — use the tabs for demand forecast, ADR, cancellation risk, LOS, channel mix, diagnostics, feature importance, and data explorer.
6. **Score** — in the Cancellation Risk tab, use "Score a New Booking" to get cancellation probability and predicted LOS.

## Model Validation Methodology

### Walk-Forward Cross-Validation

Instead of a single train-test split, models are evaluated using expanding-window walk-forward CV:
- Training window grows over time (simulating real-world usage)
- Multiple test periods are evaluated (typically 5 folds of 30 days each)
- Metrics (R², MAE, MAPE) are computed per fold
- **Robust R²** = median R² across folds (reported as primary quality metric)
- **Expected R² Range** = [min, max] R² across folds (shows expected variance)

### Feature Pruning

Features are pruned based on importance stability across CV folds:
- Features with high variance in importance (unstable) are removed
- Low-importance features that are also unstable are pruned
- Time-based features are always retained (essential for seasonality)
- This reduces overfitting and improves generalization

### Drift Detection

When new data is uploaded, the system compares it to the training baseline:
- **Numeric drift**: Cohen's d (standardized mean difference)
- **Categorical drift**: Jensen-Shannon distance
- **Overall drift score**: Aggregated across all features
- **Drift levels**: low / medium / high

High drift triggers warnings and may indicate that retraining is needed.

### Model Promotion Criteria

New models are only promoted if:
1. Robust R² does not drop significantly (> 0.02 below previous)
2. R² variance across folds is reasonable (< 0.15 std)
3. MAPE does not increase significantly (> 2% above previous)
4. Robust R² meets minimum threshold (≥ 0.30)

If promotion fails, previous models are retained and reasons are displayed.

## Interpreting Model Quality

The sidebar shows **Robust R²** with the expected range in parentheses:

```
Occupancy % (Overall) R²=0.823 (range: 0.78-0.88) MAPE=5.2%
```

This means:
- Median out-of-sample R² across validation folds is 0.823
- R² typically falls between 0.78 and 0.88 on new data
- If your new data is similar to validation periods, expect ~0.82 R²
- If patterns shift significantly, R² could fall toward 0.78 or lower

## Project structure

```
Demand-forecast/
├── hotel_demand_forecast.py   # Streamlit app and all models
├── requirements.txt
├── README.md
└── .gitignore
```

## License

Use and modify as needed for your organization.
