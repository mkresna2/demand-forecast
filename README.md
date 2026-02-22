# Hotel Revenue Intelligence Platform

A Streamlit app for hotel demand forecasting and revenue intelligence. Train LightGBM models on your bookings data to forecast revenue, occupancy, ADR, channel mix, and to score cancellation risk and length-of-stay for individual bookings.

## Features

- **Time-series models (stay-date level)**  
  Nightly revenue, overall and room-type occupancy %, ADR, and channel mix (Direct, OTA, Walk-in, Website) — all forecast with LightGBM regressors.

- **Booking-level models**  
  Cancellation probability (LightGBM classifier) and length-of-stay prediction (LightGBM regressor).

- **Interactive dashboard**  
  Tabs for demand forecast, ADR forecast, cancellation risk, length-of-stay, channel mix, diagnostics, feature importance, and data explorer. Configurable forecast horizon (7–180 days) and room capacity.

- **Single-booking scoring**  
  Enter check-in/out, channel, rate plan, and get cancellation probability and predicted LOS with risk tier and recommendations.

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
2. **Train** — click **Train All Models** to build the 12 LightGBM models.
3. **Explore** — use the tabs for demand forecast, ADR, cancellation risk, LOS, channel mix, diagnostics, feature importance, and data explorer.
4. **Score** — in the Cancellation Risk tab, use “Score a New Booking” to get cancellation probability and predicted LOS for a single booking.

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
