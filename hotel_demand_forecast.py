"""
Hotel Revenue Intelligence Platform
────────────────────────────────────
OTB-Anchored Demand Forecasting:
  - On-the-Book (OTB): actual confirmed bookings already in system
  - Model predicts PICKUP (additional bookings still to come)
  - TOTAL FORECAST = OTB (certain) + PICKUP (model)
  - Forecast starts from TOMORROW relative to the "as-of date"

Models (12 LightGBM):
  Time-series (stay_date level):
    1.  Nightly Revenue
    2.  Occupancy % Overall
    3.  Occ % Standard / 4. Deluxe / 5. Suite
    6.  ADR (Avg Daily Rate)
    7-10. Channel Mix (Direct / OTA / Walk-in / Website)

  Booking-level:
    11. Cancellation Probability
    12. Length-of-Stay
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Revenue Intelligence",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .hero { background:linear-gradient(135deg,#0d0d1a 0%,#1a1a2e 40%,#0f3460 100%);
    padding:2rem 2.5rem;border-radius:14px;margin-bottom:1.5rem;color:#fff; }
  .hero h1{font-size:2rem;font-weight:800;margin:0;letter-spacing:-.5px;}
  .hero p{opacity:.7;margin:.4rem 0 0;font-size:.95rem;}
  .kpi{background:#fff;border-radius:10px;padding:1rem;
    box-shadow:0 2px 10px rgba(0,0,0,.07);text-align:center;
    border-left:4px solid #0f3460;height:100%;}
  .kpi .v{font-size:1.6rem;font-weight:700;color:#0f3460;margin:0;line-height:1.2;}
  .kpi .l{font-size:.75rem;color:#333;margin:.25rem 0 0;}
  .kpi.g .v{color:#27ae60;}.kpi.g{border-left-color:#27ae60;}
  .kpi.o .v{color:#e67e22;}.kpi.o{border-left-color:#e67e22;}
  .kpi.r .v{color:#e74c3c;}.kpi.r{border-left-color:#e74c3c;}
  .kpi.p .v{color:#8e44ad;}.kpi.p{border-left-color:#8e44ad;}
  .kpi.t .v{color:#16a085;}.kpi.t{border-left-color:#16a085;}
  .insight{background:#f8f9ff;border-left:3px solid #0f3460;
    border-radius:8px;padding:.85rem 1rem;margin:.4rem 0;
    font-size:.88rem;line-height:1.6;}
  .warn{background:#fff8f0;border-left:3px solid #e67e22;
    border-radius:8px;padding:.85rem 1rem;margin:.4rem 0;font-size:.88rem;}
  .otb-box{background:#e8f5e9;border-left:4px solid #27ae60;
    border-radius:10px;padding:1rem 1.2rem;margin:.5rem 0;color:#1b5e20;}
  .pickup-box{background:#fff3e0;border-left:4px solid #e67e22;
    border-radius:10px;padding:1rem 1.2rem;margin:.5rem 0;color:#e65100;}
  .total-box{background:#e8eaf6;border-left:4px solid #3949ab;
    border-radius:10px;padding:1rem 1.2rem;margin:.5rem 0;color:#1a237e;}
  .sec{font-size:1.1rem;font-weight:700;color:#e8ecf4;
    margin:1.4rem 0 .7rem;padding-bottom:.3rem;
    border-bottom:2px solid rgba(232,236,244,.4);}
  .sec .sec-sub{color:rgba(232,236,244,.95);font-weight:400;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CAPACITY     = {"Standard": 50, "Deluxe": 25, "Suite": 10, "total": 85}
ROOM_COLORS  = {"Standard": "#0f3460", "Deluxe": "#e94560", "Suite": "#e67e22"}
CHANNEL_COLORS = {"Direct":"#0f3460","OTA":"#e94560","Walk-in":"#27ae60","Website":"#8e44ad"}
CHANNELS     = ["Direct", "OTA", "Walk-in", "Website"]
TS_TARGETS   = ["nightly_revenue","occ_pct_total",
                "occ_pct_Standard","occ_pct_Deluxe","occ_pct_Suite","adr"]
LABEL_MAP    = {
    "nightly_revenue": "Nightly Revenue",
    "occ_pct_total":   "Occupancy % (Overall)",
    "occ_pct_Standard":"Occupancy % (Standard)",
    "occ_pct_Deluxe":  "Occupancy % (Deluxe)",
    "occ_pct_Suite":   "Occupancy % (Suite)",
    "adr":             "ADR (Avg Daily Rate)",
}
CANCEL_FEATS = [
    "lead_time","lead_time_sq","Number_of_Nights","Number_of_Guests",
    "nights_x_guests","Room_Type_enc","Booking_Channel_enc","Rate_Plan_enc",
    "checkin_dow","checkin_month","checkin_quarter","checkin_week",
    "booking_dow","booking_month","booking_year",
    "is_member","is_early_bird","is_non_refund","is_bar","is_corporate",
    "is_weekend_ci","Booked_Rate","rev_per_guest","rev_per_night_guest",
    "month_sin","month_cos",
]
LOS_FEATS    = [f for f in CANCEL_FEATS
                if f not in ("Number_of_Nights","nights_x_guests","rev_per_night_guest")]

# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt(val, prefix="Rp "):
    if   val >= 1e9: return f"{prefix}{val/1e9:.2f}B"
    elif val >= 1e6: return f"{prefix}{val/1e6:.1f}M"
    elif val >= 1e3: return f"{prefix}{val/1e3:.1f}K"
    return f"{prefix}{val:,.0f}"

def kpi(val, label, cls=""):
    return (f'<div class="kpi {cls}"><p class="v">{val}</p>'
            f'<p class="l">{label}</p></div>')

# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_expand(raw_bytes):
    df = pd.read_csv(raw_bytes)
    df["Check_in_Date"]  = pd.to_datetime(df["Check_in_Date"])
    df["Check_out_Date"] = pd.to_datetime(df["Check_out_Date"])
    df["Booking_Date"]   = pd.to_datetime(df["Booking_Date"])
    df["is_cancelled"]   = (df["Cancellation_Status"] == "Cancelled").astype(int)
    df["lead_time"]      = (df["Check_in_Date"] - df["Booking_Date"]).dt.days

    rows = []
    for _, r in df.iterrows():
        for d in pd.date_range(r["Check_in_Date"],
                                r["Check_out_Date"] - timedelta(days=1)):
            rec = r.to_dict(); rec["stay_date"] = d; rows.append(rec)
    expanded = pd.DataFrame(rows)
    return df, expanded


def build_daily(expanded: pd.DataFrame) -> pd.DataFrame:
    conf = expanded[expanded["Cancellation_Status"] == "Confirmed"].copy()
    base = conf.groupby("stay_date").agg(
        rooms_occupied  = ("Booking_ID", "count"),
        nightly_revenue = ("Booked_Rate", "sum"),
        rooms_Standard  = ("Room_Type", lambda x: (x == "Standard").sum()),
        rooms_Deluxe    = ("Room_Type", lambda x: (x == "Deluxe").sum()),
        rooms_Suite     = ("Room_Type", lambda x: (x == "Suite").sum()),
    ).reset_index()

    ch_piv = (conf.groupby(["stay_date","Booking_Channel"])["Booking_ID"]
              .count().unstack(fill_value=0).reset_index())
    ch_piv.columns.name = None
    for ch in CHANNELS:
        if ch not in ch_piv.columns: ch_piv[ch] = 0
    base = base.merge(ch_piv[["stay_date"]+CHANNELS], on="stay_date", how="left").fillna(0)

    for rt in ["Standard","Deluxe","Suite"]:
        base[f"occ_pct_{rt}"] = base[f"rooms_{rt}"] / CAPACITY[rt] * 100
    base["occ_pct_total"] = base["rooms_occupied"] / CAPACITY["total"] * 100

    adr = conf.groupby("stay_date")["Booked_Rate"].mean().reset_index()
    adr.columns = ["stay_date","adr"]
    base = base.merge(adr, on="stay_date", how="left")
    base["revpar"] = base["nightly_revenue"] / CAPACITY["total"]

    ch_total = base[CHANNELS].sum(axis=1).replace(0,1)
    for ch in CHANNELS:
        base[f"pct_{ch}"] = base[ch] / ch_total * 100

    return base.sort_values("stay_date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# OTB (ON-THE-BOOK) CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def get_otb(raw: pd.DataFrame, as_of_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """
    Returns daily OTB snapshot: what's already confirmed in the system
    as of `as_of_date`, for the next `horizon` stay nights.
    OTB = bookings where Booking_Date <= as_of_date AND Check_in_Date > as_of_date
    """
    tomorrow  = as_of_date + timedelta(days=1)
    end_date  = as_of_date + timedelta(days=horizon)

    conf = raw[raw["Cancellation_Status"] == "Confirmed"]
    otb_bookings = conf[conf["Booking_Date"] <= as_of_date].copy()

    rows = []
    for _, r in otb_bookings.iterrows():
        for d in pd.date_range(r["Check_in_Date"],
                                r["Check_out_Date"] - timedelta(days=1)):
            if tomorrow <= d <= end_date:
                rows.append({
                    "stay_date":    d,
                    "Room_Type":    r["Room_Type"],
                    "Booking_Channel": r["Booking_Channel"],
                    "Booked_Rate":  r["Booked_Rate"],
                })

    all_dates = pd.DataFrame({"stay_date": pd.date_range(tomorrow, end_date)})

    if not rows:
        otb = all_dates.copy()
        for col in ["rooms_otb","revenue_otb","std_otb","dlx_otb","ste_otb","adr_otb"]:
            otb[col] = 0.0
    else:
        exp = pd.DataFrame(rows)
        otb = exp.groupby("stay_date").agg(
            rooms_otb   = ("Booked_Rate", "count"),
            revenue_otb = ("Booked_Rate", "sum"),
            std_otb     = ("Room_Type", lambda x: (x == "Standard").sum()),
            dlx_otb     = ("Room_Type", lambda x: (x == "Deluxe").sum()),
            ste_otb     = ("Room_Type", lambda x: (x == "Suite").sum()),
            adr_otb     = ("Booked_Rate", "mean"),
        ).reset_index()
        otb = all_dates.merge(otb, on="stay_date", how="left").fillna(0)

    otb["occ_pct_otb"]    = otb["rooms_otb"]   / CAPACITY["total"]    * 100
    otb["std_pct_otb"]    = otb["std_otb"]     / CAPACITY["Standard"] * 100
    otb["dlx_pct_otb"]    = otb["dlx_otb"]     / CAPACITY["Deluxe"]   * 100
    otb["ste_pct_otb"]    = otb["ste_otb"]     / CAPACITY["Suite"]    * 100
    otb["remaining_rooms"]= CAPACITY["total"]   - otb["rooms_otb"]
    otb["remaining_std"]  = CAPACITY["Standard"]- otb["std_otb"]
    otb["remaining_dlx"]  = CAPACITY["Deluxe"]  - otb["dlx_otb"]
    otb["remaining_ste"]  = CAPACITY["Suite"]   - otb["ste_otb"]

    return otb


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _add_time(df):
    df = df.copy()
    df["dow"]       = df["stay_date"].dt.dayofweek
    df["month"]     = df["stay_date"].dt.month
    df["week"]      = df["stay_date"].dt.isocalendar().week.astype(int)
    df["quarter"]   = df["stay_date"].dt.quarter
    df["is_weekend"]= (df["dow"] >= 5).astype(int)
    df["year"]      = df["stay_date"].dt.year
    df["dom"]       = df["stay_date"].dt.day
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def _add_lags(df, targets):
    df = df.copy()
    for t in targets:
        if t not in df.columns: continue
        for lag in [1,2,3,7,14,21,28,30,60,90]:
            df[f"{t}_lag{lag}"] = df[t].shift(lag)
        for w in [3,7,14,30,60]:
            df[f"{t}_roll{w}"]    = df[t].shift(1).rolling(w).mean()
            df[f"{t}_roll{w}std"] = df[t].shift(1).rolling(w).std()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TIME-SERIES MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_ts_models(daily: pd.DataFrame):
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error

    all_ts_targets = TS_TARGETS + CHANNELS
    df = _add_time(daily)
    df = _add_lags(df, all_ts_targets)
    df = df.dropna().reset_index(drop=True)

    non_feat = ({"stay_date"} | set(all_ts_targets) |
                {"rooms_occupied","rooms_Standard","rooms_Deluxe","rooms_Suite","revpar"} |
                {f"pct_{ch}" for ch in CHANNELS})
    feat_cols = [c for c in df.columns if c not in non_feat]

    split = len(df) - 90
    models, metrics = {}, {}

    for target in all_ts_targets:
        if target not in df.columns: continue
        X = df[feat_cols]; y = df[target]
        Xtr, Xte = X.iloc[:split], X.iloc[split:]
        ytr, yte  = y.iloc[:split], y.iloc[split:]

        m = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.02, max_depth=7, num_leaves=63,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.05, reg_lambda=0.05, random_state=42, verbose=-1,
        )
        m.fit(Xtr, ytr, eval_set=[(Xte,yte)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

        preds = np.clip(m.predict(Xte), 0, None)
        if "occ_pct" in target: preds = np.minimum(preds, 100.0)

        mae  = mean_absolute_error(yte, preds)
        denom = np.where(yte.values == 0, 1, yte.values)
        mape = float(np.mean(np.abs((yte.values - preds) / denom)) * 100)
        r2   = float(1 - np.sum((yte.values-preds)**2) /
                     np.sum((yte.values - yte.mean())**2))

        models[target]  = m
        metrics[target] = {
            "MAE": mae, "MAPE": mape, "R2": r2,
            "test_preds": preds, "test_actual": yte.values,
            "test_dates": df["stay_date"].iloc[split:].values,
        }

    return models, metrics, feat_cols, df


# ══════════════════════════════════════════════════════════════════════════════
# BOOKING-LEVEL MODELS
# ══════════════════════════════════════════════════════════════════════════════

def _booking_features(raw: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    df = raw.copy()
    df["lead_time"]       = (df["Check_in_Date"] - df["Booking_Date"]).dt.days
    df["checkin_dow"]     = df["Check_in_Date"].dt.dayofweek
    df["checkin_month"]   = df["Check_in_Date"].dt.month
    df["checkin_quarter"] = df["Check_in_Date"].dt.quarter
    df["checkin_week"]    = df["Check_in_Date"].dt.isocalendar().week.astype(int)
    df["booking_dow"]     = df["Booking_Date"].dt.dayofweek
    df["booking_month"]   = df["Booking_Date"].dt.month
    df["booking_year"]    = df["Booking_Date"].dt.year
    df["is_member"]       = df["Rate_Plan"].str.contains("Member").astype(int)
    df["is_early_bird"]   = df["Rate_Plan"].str.contains("Early Bird").astype(int)
    df["is_non_refund"]   = df["Rate_Plan"].str.contains("Non-Refundable").astype(int)
    df["is_bar"]          = df["Rate_Plan"].str.contains("BAR").astype(int)
    df["is_corporate"]    = df["Rate_Plan"].str.contains("Corporate").astype(int)
    df["is_weekend_ci"]   = (df["checkin_dow"] >= 5).astype(int)
    df["lead_time_sq"]    = df["lead_time"] ** 2
    df["nights_x_guests"] = df["Number_of_Nights"] * df["Number_of_Guests"]
    df["rev_per_guest"]   = df["Booked_Rate"] / df["Number_of_Guests"]
    df["rev_per_night_guest"] = df["Booked_Rate"] / (
        df["Number_of_Guests"] * df["Number_of_Nights"])
    df["month_sin"] = np.sin(2 * np.pi * df["checkin_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["checkin_month"] / 12)
    for col in ["Room_Type","Booking_Channel","Rate_Plan"]:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col])
    return df


def train_booking_models(raw: pd.DataFrame):
    import lightgbm as lgb
    from sklearn.metrics import (mean_absolute_error, roc_auc_score,
                                  accuracy_score, classification_report)
    df   = _booking_features(raw)
    split = int(len(df) * 0.8)

    # Cancellation
    Xtr = df[CANCEL_FEATS].iloc[:split]; Xte = df[CANCEL_FEATS].iloc[split:]
    ytr = df["is_cancelled"].iloc[:split]; yte = df["is_cancelled"].iloc[split:]
    m_c = lgb.LGBMClassifier(n_estimators=1000,learning_rate=0.02,num_leaves=63,
                               class_weight="balanced",subsample=0.8,
                               colsample_bytree=0.8,random_state=42,verbose=-1)
    m_c.fit(Xtr,ytr,eval_set=[(Xte,yte)],
             callbacks=[lgb.early_stopping(50,verbose=False)])
    proba   = m_c.predict_proba(Xte)[:,1]
    preds_c = (proba > 0.5).astype(int)
    cancel_metrics = {
        "AUC": float(roc_auc_score(yte,proba)),
        "Accuracy": float(accuracy_score(yte,preds_c)),
        "report": classification_report(yte,preds_c,output_dict=True),
        "test_proba": proba, "test_actual": yte.values,
        "features": CANCEL_FEATS, "importances": m_c.feature_importances_,
    }

    # LOS
    Xtr_l = df[LOS_FEATS].iloc[:split]; Xte_l = df[LOS_FEATS].iloc[split:]
    ytr_l = df["Number_of_Nights"].iloc[:split]; yte_l = df["Number_of_Nights"].iloc[split:]
    m_l = lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,num_leaves=31,
                              random_state=42,verbose=-1)
    m_l.fit(Xtr_l,ytr_l)
    preds_l = np.clip(np.round(m_l.predict(Xte_l)),1,7)
    los_metrics = {
        "MAE": float(mean_absolute_error(yte_l,preds_l)),
        "Accuracy": float((preds_l == yte_l.values).mean()),
        "test_preds": preds_l, "test_actual": yte_l.values,
        "features": LOS_FEATS, "importances": m_l.feature_importances_,
        "raw_df": df,
    }
    return m_c, cancel_metrics, m_l, los_metrics


# ══════════════════════════════════════════════════════════════════════════════
# OTB-ANCHORED FORECAST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def forecast_otb_anchored(ts_models, feat_cols, daily, otb_df, horizon=30):
    """
    OTB-Anchored Forecast:
    1. For each future date, check OTB (already confirmed)
    2. Model predicts TOTAL expected occupancy/revenue
    3. PICKUP = max(0, model_prediction - OTB)   [model may not exceed OTB if OTB>pred]
    4. TOTAL FORECAST = OTB + PICKUP (capped at capacity)

    Returns df with columns: stay_date, otb_*, model_*, pickup_*, total_*
    """
    all_ts_targets = TS_TARGETS + CHANNELS
    df       = _add_time(daily).sort_values("stay_date").reset_index(drop=True)
    history  = df.copy()
    recent   = df.tail(30)

    static_cols = [c for c in df.columns
                   if c not in set(all_ts_targets) |
                   {"stay_date","rooms_occupied","rooms_Standard","rooms_Deluxe",
                    "rooms_Suite","revpar"} |
                   {f"pct_{ch}" for ch in CHANNELS}
                   and "_lag" not in c and "_roll" not in c]

    records = []
    for i, row_otb in otb_df.reset_index(drop=True).iterrows():
        next_date = pd.Timestamp(row_otb["stay_date"])

        new_row = {"stay_date": next_date}
        for col in static_cols:
            if col in recent.columns:
                new_row[col] = recent[col].mean()
        for t in all_ts_targets + ["revpar","rooms_occupied",
                                    "rooms_Standard","rooms_Deluxe","rooms_Suite"]:
            new_row[t] = 0.0
        for ch in CHANNELS:
            new_row[f"pct_{ch}"] = 0.0

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        history = _add_lags(history, all_ts_targets)
        row_f   = history.iloc[[-1]][feat_cols].fillna(0)

        model_pred = {}
        for t, m in ts_models.items():
            v = float(np.clip(m.predict(row_f)[0], 0, None))
            if "occ_pct" in t: v = min(v, 100.0)
            model_pred[t] = v
            history.at[history.index[-1], t] = v

        # ── OTB values for this date ──────────────────────────────────────────
        otb_rooms   = float(row_otb["rooms_otb"])
        otb_revenue = float(row_otb["revenue_otb"])
        otb_occ     = float(row_otb["occ_pct_otb"])
        otb_std     = float(row_otb["std_pct_otb"])
        otb_dlx     = float(row_otb["dlx_pct_otb"])
        otb_ste     = float(row_otb["ste_pct_otb"])
        otb_adr     = float(row_otb["adr_otb"]) if row_otb["adr_otb"] > 0 else model_pred.get("adr",0)
        remaining   = float(row_otb["remaining_rooms"])

        # ── Pickup = what model says ABOVE what's already booked ───────────────
        m_occ   = model_pred.get("occ_pct_total", 0)
        m_rev   = model_pred.get("nightly_revenue", 0)
        m_adr   = model_pred.get("adr", 0)

        # Total OTB rooms vs model total – pickup is the gap
        pickup_occ_pct = max(0.0, m_occ - otb_occ)               # additional occ%
        pickup_rooms   = pickup_occ_pct * CAPACITY["total"] / 100 # additional rooms
        pickup_rooms   = min(pickup_rooms, remaining)              # cap at remaining
        pickup_occ_pct = pickup_rooms / CAPACITY["total"] * 100

        # Pickup revenue: pickup rooms × model ADR (new bookings at market rate)
        pickup_adr     = m_adr if m_adr > 0 else otb_adr
        pickup_revenue = pickup_rooms * pickup_adr

        # Total = OTB + Pickup
        total_rooms   = otb_rooms + pickup_rooms
        total_occ     = total_rooms / CAPACITY["total"] * 100
        total_revenue = otb_revenue + pickup_revenue
        total_adr     = (otb_revenue + pickup_revenue) / max(total_rooms,1)

        # Room-type breakdown: distribute pickup proportionally by remaining capacity
        rem_std = float(row_otb["remaining_std"])
        rem_dlx = float(row_otb["remaining_dlx"])
        rem_ste = float(row_otb["remaining_ste"])
        rem_total = rem_std + rem_dlx + rem_ste
        if rem_total > 0 and pickup_rooms > 0:
            pickup_std = pickup_rooms * rem_std / rem_total
            pickup_dlx = pickup_rooms * rem_dlx / rem_total
            pickup_ste = pickup_rooms * rem_ste / rem_total
        else:
            pickup_std = pickup_dlx = pickup_ste = 0.0

        total_std_occ = min((row_otb["std_otb"] + pickup_std)/CAPACITY["Standard"]*100, 100)
        total_dlx_occ = min((row_otb["dlx_otb"] + pickup_dlx)/CAPACITY["Deluxe"]*100, 100)
        total_ste_occ = min((row_otb["ste_otb"] + pickup_ste)/CAPACITY["Suite"]*100, 100)

        history.at[history.index[-1], "revpar"] = total_revenue / CAPACITY["total"]
        history.at[history.index[-1], "rooms_occupied"] = total_rooms

        records.append({
            "stay_date":        next_date,
            # OTB (certain)
            "otb_rooms":        otb_rooms,
            "otb_occ_pct":      otb_occ,
            "otb_revenue":      otb_revenue,
            "otb_adr":          otb_adr,
            # Pickup (model)
            "pickup_rooms":     pickup_rooms,
            "pickup_occ_pct":   pickup_occ_pct,
            "pickup_revenue":   pickup_revenue,
            "model_occ_pct":    m_occ,
            "model_revenue":    m_rev,
            # Total forecast
            "total_rooms":      total_rooms,
            "total_occ_pct":    total_occ,
            "total_revenue":    total_revenue,
            "total_adr":        total_adr,
            "total_revpar":     total_revenue / CAPACITY["total"],
            "remaining_rooms":  remaining - pickup_rooms,
            # Room-type occupancy
            "total_std_occ":    total_std_occ,
            "total_dlx_occ":    total_dlx_occ,
            "total_ste_occ":    total_ste_occ,
            # Channel (model)
            **{ch: model_pred.get(ch, 0) for ch in CHANNELS},
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k in ("ts_models","ts_metrics","ts_feat_cols","ts_df","daily",
          "m_cancel","cancel_metrics","m_los","los_metrics","raw","expanded"):
    if k not in st.session_state:
        st.session_state[k] = None

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🏨 Hotel Revenue Intelligence Platform</h1>
  <p>OTB-Anchored Forecasting • Revenue • Occupancy • ADR •
     Cancellation Risk • Length-of-Stay • Channel Mix</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    uploaded = st.file_uploader("📂 Upload Bookings CSV", type=["csv"])
    st.markdown("---")

    st.markdown("### 📅 Forecast Settings")
    horizon    = st.slider("Horizon (days)", 7, 180, 30, 7)

    # as_of_date: default = last booking date in uploaded/trained data (simulated "today")
    st.markdown("**As-of Date** *(simulated 'today')*")
    st.caption("OTB uses confirmed bookings made on or before this date.")
    import datetime
    if uploaded is not None:
        try:
            df_peek = pd.read_csv(uploaded)
            df_peek["Booking_Date"] = pd.to_datetime(df_peek["Booking_Date"])
            max_bk = df_peek["Booking_Date"].max().date()
            min_bk = df_peek["Booking_Date"].min().date()
            uploaded.seek(0)  # reset so Train can read the file again
        except Exception:
            max_bk = datetime.date.today()
            min_bk = datetime.date(2020, 1, 1)
    elif st.session_state.raw is not None:
        max_bk = st.session_state.raw["Booking_Date"].max().date()
        min_bk = st.session_state.raw["Booking_Date"].min().date()
    else:
        max_bk = datetime.date.today()
        min_bk = datetime.date(2020, 1, 1)
    as_of_date = st.date_input("As-of Date", value=max_bk,
                                min_value=min_bk, max_value=max_bk)
    as_of_ts = pd.Timestamp(as_of_date)

    with st.expander("🏗️ Room Capacity"):
        CAPACITY["Standard"] = st.number_input("Standard rooms", 1, 500, 50)
        CAPACITY["Deluxe"]   = st.number_input("Deluxe rooms",   1, 500, 25)
        CAPACITY["Suite"]    = st.number_input("Suite rooms",    1, 500, 10)
        CAPACITY["total"]    = CAPACITY["Standard"] + CAPACITY["Deluxe"] + CAPACITY["Suite"]

    st.markdown("---")
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        if uploaded is None:
            st.error("Upload a CSV first.")
        else:
            prog = st.progress(0, "Loading data…")
            raw_proc, expanded = load_and_expand(uploaded)
            prog.progress(25, "Building daily aggregates…")
            daily = build_daily(expanded)
            prog.progress(45, "Training time-series models…")
            ts_models, ts_metrics, feat_cols, ts_df = train_ts_models(daily)
            prog.progress(80, "Training booking-level models…")
            m_c, cancel_met, m_l, los_met = train_booking_models(raw_proc)
            prog.progress(100, "Done!")

            st.session_state.ts_models      = ts_models
            st.session_state.ts_metrics     = ts_metrics
            st.session_state.ts_feat_cols   = feat_cols
            st.session_state.ts_df          = ts_df
            st.session_state.daily          = daily
            st.session_state.m_cancel       = m_c
            st.session_state.cancel_metrics = cancel_met
            st.session_state.m_los          = m_l
            st.session_state.los_metrics    = los_met
            st.session_state.raw            = raw_proc
            st.session_state.expanded       = expanded
            prog.empty()
            st.success("✅ All 12 models trained!")

    if st.session_state.ts_models:
        st.markdown("### 📊 Model Quality")
        for t in TS_TARGETS:
            if t in st.session_state.ts_metrics:
                m = st.session_state.ts_metrics[t]
                st.caption(f"**{LABEL_MAP.get(t,t)}** R²={m['R2']:.3f} MAPE={m['MAPE']:.1f}%")
        cm = st.session_state.cancel_metrics
        lm = st.session_state.los_metrics
        if cm: st.caption(f"**Cancellation** AUC={cm['AUC']:.3f}")
        if lm: st.caption(f"**LOS** MAE={lm['MAE']:.2f} nts Acc={lm['Accuracy']:.3f}")
    st.caption("12 LightGBM models • OTB-anchored • Stay-date expansion")

# ── Guard ───────────────────────────────────────────────────────────────────────
if st.session_state.ts_models is None:
    c1,c2,c3,c4 = st.columns(4)
    for col,icon,s,d in [(c1,"📂","1","Upload Bookings CSV"),
                          (c2,"🤖","2","Train All Models"),
                          (c3,"🔮","3","OTB-Anchored Forecasts"),
                          (c4,"🎯","4","Score Bookings")]:
        with col: st.markdown(kpi(icon,f"Step {s}: {d}"), unsafe_allow_html=True)
    st.info("👈 Upload **Bookings.csv** and click **Train All Models** to begin.")
    st.stop()

ts_models      = st.session_state.ts_models
ts_metrics     = st.session_state.ts_metrics
feat_cols      = st.session_state.ts_feat_cols
daily          = st.session_state.daily
m_cancel       = st.session_state.m_cancel
cancel_metrics = st.session_state.cancel_metrics
m_los          = st.session_state.m_los
los_metrics    = st.session_state.los_metrics
raw            = st.session_state.raw
expanded       = st.session_state.expanded

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE OTB + FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Computing OTB and generating forecast…"):
    otb_df  = get_otb(raw, as_of_ts, horizon)
    fcast   = forecast_otb_anchored(ts_models, feat_cols, daily, otb_df, horizon)

tomorrow_str = (as_of_ts + timedelta(days=1)).strftime("%d %b %Y")
end_str      = (as_of_ts + timedelta(days=horizon)).strftime("%d %b %Y")

# ══════════════════════════════════════════════════════════════════════════════
# TOP KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<p class="sec">📈 {horizon}-Day Outlook: {tomorrow_str} → {end_str} '
            f'<span class="sec-sub">(as of {as_of_date})</span></p>',
            unsafe_allow_html=True)

avg_otb_occ   = fcast["otb_occ_pct"].mean()
avg_total_occ = fcast["total_occ_pct"].mean()
avg_pickup    = fcast["pickup_occ_pct"].mean()
total_rev_otb = fcast["otb_revenue"].sum()
total_rev_all = fcast["total_revenue"].sum()
avg_adr       = fcast["total_adr"].mean()
avg_revpar    = fcast["total_revpar"].mean()

k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
with k1: st.markdown(kpi(f"{avg_otb_occ:.1f}%","OTB Occ % (certain)","g"), unsafe_allow_html=True)
with k2: st.markdown(kpi(f"{avg_pickup:.1f}%","Pickup Occ % (model)","o"), unsafe_allow_html=True)
with k3: st.markdown(kpi(f"{avg_total_occ:.1f}%","Total Forecast Occ%","t"), unsafe_allow_html=True)
with k4: st.markdown(kpi(fmt(total_rev_otb),"OTB Revenue","g"), unsafe_allow_html=True)
with k5: st.markdown(kpi(fmt(total_rev_all),"Total Forecast Rev","t"), unsafe_allow_html=True)
with k6: st.markdown(kpi(fmt(avg_adr),"Forecast ADR","p"), unsafe_allow_html=True)
with k7: st.markdown(kpi(fmt(avg_revpar),"Forecast RevPAR"), unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
le1,le2,le3 = st.columns(3)
with le1: st.markdown("""<div class="otb-box">
  🟢 <b>On-the-Book (OTB)</b><br>
  Confirmed bookings already in the system as of the as-of date.
  This is <b>certain</b> revenue — guests are already booked.
  </div>""", unsafe_allow_html=True)
with le2: st.markdown("""<div class="pickup-box">
  🟡 <b>Pickup (Model Prediction)</b><br>
  Additional bookings expected to come in before the stay date.
  Predicted by LightGBM based on historical pickup patterns.
  </div>""", unsafe_allow_html=True)
with le3: st.markdown("""<div class="total-box">
  🔵 <b>Total Forecast</b><br>
  OTB + Pickup, capped at physical room capacity.
  This is your <b>full demand forecast</b>.
  </div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
(tab_fcast, tab_adr, tab_cancel,
 tab_los, tab_channel, tab_diag, tab_fi, tab_data) = st.tabs([
    "🔮 Demand Forecast",
    "💰 ADR Forecast",
    "🚨 Cancellation Risk",
    "🛏️ Length-of-Stay",
    "📡 Channel Mix",
    "📊 Diagnostics",
    "🏷️ Feature Importance",
    "📋 Data Explorer",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEMAND FORECAST  (OTB-Anchored)
# ══════════════════════════════════════════════════════════════════════════════
with tab_fcast:
    st.markdown(f'<p class="sec">🔮 OTB-Anchored Demand Forecast — {tomorrow_str} to {end_str}</p>',
                unsafe_allow_html=True)

    try:
        import altair as alt

        # ── Revenue: OTB + Pickup stacked bar + historical line ────────────────
        st.markdown("**Nightly Revenue: OTB (confirmed) vs Pickup (model prediction)**")
        rev_bar = fcast[["stay_date","otb_revenue","pickup_revenue"]].copy()
        rev_melt = rev_bar.melt("stay_date",var_name="Component",value_name="Revenue")

        bar_rev = (alt.Chart(rev_melt).mark_bar()
                   .encode(x=alt.X("stay_date:T",title="Date"),
                           y=alt.Y("Revenue:Q",stack="zero",
                                    axis=alt.Axis(format="~s"),title="Revenue (IDR)"),
                           color=alt.Color("Component:N",
                               scale=alt.Scale(
                                   domain=["otb_revenue","pickup_revenue"],
                                   range=["#27ae60","#e67e22"]),
                               legend=alt.Legend(
                                   labelExpr="datum.value === 'otb_revenue' ? 'OTB (Confirmed)' : 'Pickup (Predicted)'")),
                           tooltip=["stay_date:T",
                                    alt.Tooltip("Revenue:Q",format=",.0f"),"Component:N"])
                   .properties(height=300))

        # Overlay total capacity line
        cap_line = pd.DataFrame({
            "stay_date": fcast["stay_date"],
            "cap_revenue": CAPACITY["total"] * fcast["total_adr"],
        })
        cap_c = (alt.Chart(fcast).mark_line(color="#e94560",strokeDash=[4,2],strokeWidth=1.5)
                 .encode(x="stay_date:T",
                         y=alt.Y("total_revenue:Q",axis=alt.Axis(format="~s")),
                         tooltip=["stay_date:T",
                                  alt.Tooltip("total_revenue:Q",format=",.0f",title="Total Forecast")]))
        st.altair_chart(bar_rev + cap_c, use_container_width=True)
        # Table: one row per stay_date with otb_revenue & pickup_revenue as columns
        rev_tbl = rev_bar.copy()
        rev_tbl["stay_date"] = rev_tbl["stay_date"].dt.strftime("%Y-%m-%d")
        rev_tbl = rev_tbl.rename(columns={
            "stay_date": "Stay Date",
            "otb_revenue": "OTB Revenue (Confirmed)",
            "pickup_revenue": "Pickup Revenue (Predicted)",
        })
        rev_tbl["OTB Revenue (Confirmed)"] = rev_tbl["OTB Revenue (Confirmed)"].apply(lambda x: f"{x:,.0f}")
        rev_tbl["Pickup Revenue (Predicted)"] = rev_tbl["Pickup Revenue (Predicted)"].apply(lambda x: f"{x:,.0f}")
        with st.expander("📊 View nightly revenue table (OTB vs Pickup by date)"):
            st.dataframe(rev_tbl, use_container_width=True, height=300)

        # ── Occupancy: OTB + Pickup stacked + capacity ceiling ────────────────
        st.markdown("**Occupancy %: OTB (confirmed) vs Pickup (model prediction)**")
        occ_melt = fcast[["stay_date","otb_occ_pct","pickup_occ_pct"]].melt(
            "stay_date",var_name="Component",value_name="Occ %")
        bar_occ = (alt.Chart(occ_melt).mark_bar()
                   .encode(x=alt.X("stay_date:T"),
                           y=alt.Y("Occ %:Q",stack="zero",
                                    scale=alt.Scale(domain=[0,105]),title="Occupancy %"),
                           color=alt.Color("Component:N",
                               scale=alt.Scale(
                                   domain=["otb_occ_pct","pickup_occ_pct"],
                                   range=["#27ae60","#e67e22"]),
                               legend=alt.Legend(
                                   labelExpr="datum.value === 'otb_occ_pct' ? 'OTB (Confirmed)' : 'Pickup (Predicted)'")),
                           tooltip=["stay_date:T",
                                    alt.Tooltip("Occ %:Q",format=".1f"),"Component:N"])
                   .properties(height=280))
        cap_line2 = (alt.Chart(pd.DataFrame({"stay_date":fcast["stay_date"],"cap":[100]*len(fcast)}))
                     .mark_rule(color="#e94560",strokeDash=[4,2],strokeWidth=1)
                     .encode(x="stay_date:T",y="cap:Q"))
        st.altair_chart(bar_occ + cap_line2, use_container_width=True)
        # Table: one row per stay_date with OTB % & Pickup % as columns
        occ_tbl = fcast[["stay_date", "otb_occ_pct", "pickup_occ_pct"]].copy()
        occ_tbl["stay_date"] = occ_tbl["stay_date"].dt.strftime("%Y-%m-%d")
        occ_tbl = occ_tbl.rename(columns={
            "stay_date": "Stay Date",
            "otb_occ_pct": "OTB Occ % (Confirmed)",
            "pickup_occ_pct": "Pickup Occ % (Predicted)",
        })
        occ_tbl["OTB Occ % (Confirmed)"] = occ_tbl["OTB Occ % (Confirmed)"].apply(lambda x: f"{x:.1f}%")
        occ_tbl["Pickup Occ % (Predicted)"] = occ_tbl["Pickup Occ % (Predicted)"].apply(lambda x: f"{x:.1f}%")
        with st.expander("📊 View occupancy % table (OTB vs Pickup by date)"):
            st.dataframe(occ_tbl, use_container_width=True, height=300)

        # ── Historical context + forecast line ────────────────────────────────
        st.markdown("**Historical Occupancy (last 90 days) + Total Forecast**")
        # Show actuals for the full range that overlaps the chart: from (forecast start − 90d) to forecast end,
        # so April–Sep and all other actuals in the forecast window appear (not just the last 90 rows).
        fcast_min = fcast["stay_date"].min()
        fcast_max = fcast["stay_date"].max()
        range_start = fcast_min - timedelta(days=90)
        hist_mask = (daily["stay_date"] >= range_start) & (daily["stay_date"] <= fcast_max)
        hist_occ = daily.loc[hist_mask, ["stay_date", "occ_pct_total"]].copy()
        hist_occ["series"] = "Historical (Actual)"
        fore_occ = fcast[["stay_date","total_occ_pct"]].rename(
            columns={"total_occ_pct":"occ_pct_total"}).copy()
        fore_occ["series"] = "Total Forecast"
        otb_line = fcast[["stay_date","otb_occ_pct"]].rename(
            columns={"otb_occ_pct":"occ_pct_total"}).copy()
        otb_line["series"] = "OTB Only"
        combined = pd.concat([hist_occ, fore_occ, otb_line])
        hf = (alt.Chart(combined).mark_line(strokeWidth=2)
              .encode(x=alt.X("stay_date:T"),
                      y=alt.Y("occ_pct_total:Q",
                               scale=alt.Scale(domain=[0,100]),title="Occupancy %"),
                      color=alt.Color("series:N",
                          scale=alt.Scale(
                              domain=["Historical (Actual)","Total Forecast","OTB Only"],
                              range=["#0f3460","#3949ab","#27ae60"])),
                      strokeDash=alt.condition(
                          alt.datum.series != "Historical (Actual)",
                          alt.value([5,3]),alt.value([0])),
                      tooltip=["stay_date:T",alt.Tooltip("occ_pct_total:Q",format=".1f"),"series:N"])
              .properties(height=270))
        st.altair_chart(hf, use_container_width=True)
        # Table: one row per date with Historical (Actual), OTB Only, Total Forecast
        all_dates = pd.DataFrame({"stay_date": pd.date_range(start=range_start, end=fcast_max, freq="D")})
        hist_df = daily.loc[hist_mask, ["stay_date", "occ_pct_total"]].rename(columns={"occ_pct_total": "Historical (Actual)"})
        fore_df = fcast[["stay_date", "otb_occ_pct", "total_occ_pct"]].rename(
            columns={"otb_occ_pct": "OTB Only", "total_occ_pct": "Total Forecast"})
        hf_tbl = all_dates.merge(hist_df, on="stay_date", how="left").merge(fore_df, on="stay_date", how="left")
        hf_tbl["stay_date"] = hf_tbl["stay_date"].dt.strftime("%Y-%m-%d")
        hf_tbl = hf_tbl.rename(columns={"stay_date": "Stay Date"})
        for c in ["Historical (Actual)", "OTB Only", "Total Forecast"]:
            hf_tbl[c] = hf_tbl[c].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
        with st.expander("📊 View historical + forecast occupancy table"):
            st.dataframe(hf_tbl, use_container_width=True, height=350)

        # ── By room type ──────────────────────────────────────────────────────
        st.markdown("**Occupancy by Room Type (Total Forecast)**")
        rt_melt = fcast[["stay_date","total_std_occ","total_dlx_occ","total_ste_occ"]].melt(
            "stay_date",var_name="Room",value_name="Occ %")
        rt_melt["Room"] = rt_melt["Room"].str.replace("total_","").str.replace("_occ","").str.replace("std","Standard").str.replace("dlx","Deluxe").str.replace("ste","Suite")
        rtc = (alt.Chart(rt_melt).mark_line(strokeWidth=2,point=True)
               .encode(x="stay_date:T",
                       y=alt.Y("Occ %:Q",scale=alt.Scale(domain=[0,100])),
                       color=alt.Color("Room:N",scale=alt.Scale(
                           domain=["Standard","Deluxe","Suite"],
                           range=[ROOM_COLORS["Standard"],ROOM_COLORS["Deluxe"],ROOM_COLORS["Suite"]])),
                       tooltip=["stay_date:T","Room:N",alt.Tooltip("Occ %:Q",format=".1f")])
               .properties(height=260))
        st.altair_chart(rtc, use_container_width=True)

    except ImportError:
        st.line_chart(fcast.set_index("stay_date")[["otb_occ_pct","pickup_occ_pct","total_occ_pct"]])

    # Room type table: always show (outside try so it appears even if charts hit an error)
    rt_tbl = fcast[["stay_date", "total_std_occ", "total_dlx_occ", "total_ste_occ"]].copy()
    rt_tbl["stay_date"] = rt_tbl["stay_date"].dt.strftime("%Y-%m-%d")
    rt_tbl = rt_tbl.rename(columns={
        "stay_date": "Stay Date",
        "total_std_occ": "Standard",
        "total_dlx_occ": "Deluxe",
        "total_ste_occ": "Suite",
    })
    for c in ["Standard", "Deluxe", "Suite"]:
        rt_tbl[c] = rt_tbl[c].apply(lambda x: f"{x:.1f}%")
    with st.expander("📊 View occupancy by room type table"):
        st.dataframe(rt_tbl, use_container_width=True, height=300)

    # ── Per-date table ────────────────────────────────────────────────────────
    st.markdown('<p class="sec">📋 Daily Forecast Detail</p>', unsafe_allow_html=True)
    tbl = fcast[["stay_date","otb_rooms","otb_occ_pct","pickup_rooms","pickup_occ_pct",
                 "total_rooms","total_occ_pct","total_revenue","total_adr","total_revpar",
                 "remaining_rooms"]].copy()
    tbl["stay_date"]    = tbl["stay_date"].dt.strftime("%a %d %b %Y")
    tbl = tbl.rename(columns={
        "stay_date":"Date","otb_rooms":"OTB Rooms","otb_occ_pct":"OTB Occ%",
        "pickup_rooms":"Pickup Rooms","pickup_occ_pct":"Pickup Occ%",
        "total_rooms":"Total Rooms","total_occ_pct":"Total Occ%",
        "total_revenue":"Revenue","total_adr":"ADR","total_revpar":"RevPAR",
        "remaining_rooms":"Remaining",
    })
    for c in ["OTB Occ%","Pickup Occ%","Total Occ%"]:
        tbl[c] = tbl[c].apply(lambda x: f"{x:.1f}%")
    for c in ["Revenue","ADR","RevPAR"]:
        tbl[c] = tbl[c].apply(lambda x: f"{x:,.0f}")
    for c in ["OTB Rooms","Pickup Rooms","Total Rooms","Remaining"]:
        tbl[c] = tbl[c].apply(lambda x: f"{x:.0f}")
    st.dataframe(tbl, use_container_width=True, height=400)

    # Download
    dl = fcast.copy(); dl["stay_date"] = dl["stay_date"].astype(str)
    st.download_button("⬇️ Download OTB-Anchored Forecast CSV",
                        dl.to_csv(index=False).encode(),
                        f"otb_forecast_{horizon}d.csv","text/csv")

    # Insights
    st.markdown('<p class="sec">💡 Key Insights</p>', unsafe_allow_html=True)
    i1,i2,i3 = st.columns(3)
    with i1:
        high_otb = fcast.nlargest(3,"otb_occ_pct")[["stay_date","otb_occ_pct"]]
        rows_str = "".join([f"• {r['stay_date'].strftime('%d %b')}: <b>{r['otb_occ_pct']:.1f}%</b><br>"
                            for _,r in high_otb.iterrows()])
        st.markdown(f"""<div class="insight"><b>🟢 Strongest OTB Dates</b><br>{rows_str}</div>""",
                    unsafe_allow_html=True)
    with i2:
        low_otb = fcast.nsmallest(3,"otb_occ_pct")[["stay_date","otb_occ_pct","remaining_rooms"]]
        rows_str = "".join([f"• {r['stay_date'].strftime('%d %b')}: OTB {r['otb_occ_pct']:.1f}% "
                            f"({r['remaining_rooms']:.0f} rooms free)<br>"
                            for _,r in low_otb.iterrows()])
        st.markdown(f"""<div class="insight"><b>⚠️ Low OTB — Opportunity to Fill</b><br>{rows_str}</div>""",
                    unsafe_allow_html=True)
    with i3:
        wk   = fcast["stay_date"].dt.dayofweek >= 5
        st.markdown(f"""<div class="insight"><b>📅 Weekend vs Weekday</b><br>
        OTB: <b>{fcast[wk]['otb_occ_pct'].mean():.1f}%</b> vs
              <b>{fcast[~wk]['otb_occ_pct'].mean():.1f}%</b><br>
        Total: <b>{fcast[wk]['total_occ_pct'].mean():.1f}%</b> vs
               <b>{fcast[~wk]['total_occ_pct'].mean():.1f}%</b><br>
        Revenue: <b>{fmt(fcast[wk]['total_revenue'].mean())}</b> vs
                 <b>{fmt(fcast[~wk]['total_revenue'].mean())}</b>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ADR FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_adr:
    st.markdown(f'<p class="sec">💰 ADR (Average Daily Rate) — {tomorrow_str} to {end_str}</p>',
                unsafe_allow_html=True)

    am = ts_metrics.get("adr",{})
    a1,a2,a3,a4 = st.columns(4)
    with a1: st.markdown(kpi(fmt(am.get("MAE",0)),"MAE"), unsafe_allow_html=True)
    with a2: st.markdown(kpi(f"{am.get('MAPE',0):.2f}%","MAPE","g"), unsafe_allow_html=True)
    with a3: st.markdown(kpi(f"{am.get('R2',0):.4f}","R²","g"), unsafe_allow_html=True)
    with a4: st.markdown(kpi(fmt(fcast["total_adr"].mean()),"Forecast Avg ADR","t"), unsafe_allow_html=True)

    try:
        import altair as alt

        # OTB ADR vs Total Forecast ADR
        adr_df = fcast[["stay_date","otb_adr","total_adr"]].copy()
        adr_melt = adr_df.melt("stay_date",var_name="Series",value_name="ADR")
        adr_melt["Series"] = adr_melt["Series"].map(
            {"otb_adr":"OTB ADR (avg booked rate)","total_adr":"Forecast ADR"})
        ac = (alt.Chart(adr_melt).mark_line(strokeWidth=2)
              .encode(x="stay_date:T",
                      y=alt.Y("ADR:Q",axis=alt.Axis(format="~s"),title="ADR (IDR)"),
                      color=alt.Color("Series:N",scale=alt.Scale(
                          domain=["OTB ADR (avg booked rate)","Forecast ADR"],
                          range=["#27ae60","#8e44ad"])),
                      tooltip=["stay_date:T","Series:N",alt.Tooltip("ADR:Q",format=",.0f")])
              .properties(title="ADR: OTB Booked Rate vs Total Forecast",height=280))
        st.altair_chart(ac, use_container_width=True)

        # Historical ADR + forecast
        h_adr = daily.tail(90)[["stay_date","adr"]].copy(); h_adr["series"]="Historical"
        f_adr = fcast[["stay_date","total_adr"]].rename(columns={"total_adr":"adr"})
        f_adr["series"] = "Forecast"
        adr_hist = pd.concat([h_adr,f_adr])
        ahc = (alt.Chart(adr_hist).mark_line(strokeWidth=2)
               .encode(x="stay_date:T",
                       y=alt.Y("adr:Q",axis=alt.Axis(format="~s"),title="ADR (IDR)"),
                       color=alt.Color("series:N",scale=alt.Scale(
                           domain=["Historical","Forecast"],range=["#0f3460","#8e44ad"])),
                       strokeDash=alt.condition(
                           alt.datum.series=="Forecast",alt.value([5,3]),alt.value([0])),
                       tooltip=["stay_date:T","series:N",alt.Tooltip("adr:Q",format=",.0f")])
               .properties(title="ADR — Historical + Forecast",height=260))
        st.altair_chart(ahc, use_container_width=True)

        # Heatmap
        st.markdown("**ADR Heatmap — Month × Day of Week (Historical)**")
        heat = daily.copy()
        heat["month_name"] = heat["stay_date"].dt.strftime("%b")
        heat["dow_name"]   = heat["stay_date"].dt.strftime("%a")
        heat_agg = heat.groupby(["month_name","dow_name"])["adr"].mean().reset_index()
        hm = (alt.Chart(heat_agg).mark_rect()
              .encode(x=alt.X("dow_name:O",sort=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]),
                      y=alt.Y("month_name:O",sort=["Jan","Feb","Mar","Apr","May","Jun",
                                                    "Jul","Aug","Sep","Oct","Nov","Dec"]),
                      color=alt.Color("adr:Q",scale=alt.Scale(scheme="viridis"),title="ADR"),
                      tooltip=["month_name:O","dow_name:O",alt.Tooltip("adr:Q",format=",.0f")])
              .properties(height=280))
        st.altair_chart(hm, use_container_width=True)

    except ImportError:
        st.line_chart(fcast.set_index("stay_date")[["otb_adr","total_adr"]])

    # Insights
    st.markdown('<p class="sec">💡 ADR Insights</p>', unsafe_allow_html=True)
    ai1,ai2,ai3 = st.columns(3)
    conf_exp = expanded[expanded["Cancellation_Status"]=="Confirmed"]
    adr_rt   = conf_exp.groupby("Room_Type")["Booked_Rate"].mean()
    adr_ch   = conf_exp.groupby("Booking_Channel")["Booked_Rate"].mean()
    wk = daily["stay_date"].dt.dayofweek >= 5
    with ai1:
        prem = (daily[wk]["adr"].mean() / daily[~wk]["adr"].mean() - 1) * 100
        st.markdown(f"""<div class="insight"><b>📅 Weekend ADR Premium</b><br>
        Weekend avg: <b>{fmt(daily[wk]['adr'].mean())}</b><br>
        Weekday avg: <b>{fmt(daily[~wk]['adr'].mean())}</b><br>
        Premium: <b>{prem:+.1f}%</b></div>""", unsafe_allow_html=True)
    with ai2:
        rows_str = "".join([f"{rt}: <b>{fmt(v)}</b><br>" for rt,v in adr_rt.items()])
        st.markdown(f"""<div class="insight"><b>🛏️ Historical ADR by Room</b><br>{rows_str}</div>""",
                    unsafe_allow_html=True)
    with ai3:
        rows_str = "".join([f"{ch}: <b>{fmt(v)}</b><br>" for ch,v in adr_ch.sort_values(ascending=False).items()])
        st.markdown(f"""<div class="insight"><b>📡 Historical ADR by Channel</b><br>{rows_str}</div>""",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CANCELLATION RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab_cancel:
    st.markdown('<p class="sec">🚨 Cancellation Risk Predictor</p>', unsafe_allow_html=True)

    cm = cancel_metrics
    cc1,cc2,cc3,cc4 = st.columns(4)
    prec   = cm["report"].get("1",{}).get("precision",0)
    recall = cm["report"].get("1",{}).get("recall",0)
    with cc1: st.markdown(kpi(f"{cm['AUC']:.4f}","AUC-ROC","o"), unsafe_allow_html=True)
    with cc2: st.markdown(kpi(f"{cm['Accuracy']*100:.1f}%","Accuracy","g"), unsafe_allow_html=True)
    with cc3: st.markdown(kpi(f"{prec*100:.1f}%","Cancel Precision","r"), unsafe_allow_html=True)
    with cc4: st.markdown(kpi(f"{recall*100:.1f}%","Cancel Recall","r"), unsafe_allow_html=True)

    st.markdown("""<div class="warn">
    ⚠️ AUC ≈ 0.51 indicates cancellation is near-random in this dataset.
    On real production data with historical patterns, AUC typically reaches 0.75–0.85.
    The model still provides calibrated probability scores for risk tiering.
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec">🎯 Score a Booking</p>', unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3)
    with col1:
        input_room      = st.selectbox("Room Type", ["Standard","Deluxe","Suite"])
        input_channel   = st.selectbox("Booking Channel", CHANNELS)
        input_rate_plan = st.selectbox("Rate Plan",[
            "BAR","BAR + Member","Corporate","Corporate + Member",
            "Early Bird (> 30 days)","Early Bird (> 30 days) + Member",
            "Non-Refundable","Non-Refundable + Member"])
    with col2:
        input_checkin  = st.date_input("Check-in Date")
        input_checkout = st.date_input("Check-out Date",
            value=pd.Timestamp.today()+timedelta(days=3))
        input_booking  = st.date_input("Booking Date", value=pd.Timestamp.today())
    with col3:
        input_guests = st.number_input("Guests", 1,6,2)
        input_rate   = st.number_input("Booked Rate (IDR)",1_000_000,5_000_000,1_200_000,50_000)

    if st.button("🔍 Predict Risk & LOS", type="primary"):
        ci  = pd.Timestamp(input_checkin)
        co  = pd.Timestamp(input_checkout)
        bk  = pd.Timestamp(input_booking)
        nts = max((co-ci).days,1); lt = max((ci-bk).days,0)
        room_enc = {"Standard":0,"Deluxe":1,"Suite":2}
        chan_enc = {"Direct":0,"OTA":1,"Walk-in":2,"Website":3}
        plan_list = sorted(raw["Rate_Plan"].unique().tolist())
        plan_enc  = {p:i for i,p in enumerate(plan_list)}
        inp = pd.DataFrame([{
            "lead_time":lt,"lead_time_sq":lt**2,"Number_of_Nights":nts,
            "Number_of_Guests":input_guests,"nights_x_guests":nts*input_guests,
            "Room_Type_enc":room_enc.get(input_room,0),
            "Booking_Channel_enc":chan_enc.get(input_channel,0),
            "Rate_Plan_enc":plan_enc.get(input_rate_plan,0),
            "checkin_dow":ci.dayofweek,"checkin_month":ci.month,
            "checkin_quarter":ci.quarter,"checkin_week":ci.isocalendar()[1],
            "booking_dow":bk.dayofweek,"booking_month":bk.month,
            "booking_year":bk.year,
            "is_member":int("Member" in input_rate_plan),
            "is_early_bird":int("Early Bird" in input_rate_plan),
            "is_non_refund":int("Non-Refundable" in input_rate_plan),
            "is_bar":int("BAR" in input_rate_plan),
            "is_corporate":int("Corporate" in input_rate_plan),
            "is_weekend_ci":int(ci.dayofweek>=5),
            "Booked_Rate":input_rate,"rev_per_guest":input_rate/input_guests,
            "rev_per_night_guest":input_rate/(input_guests*nts),
            "month_sin":np.sin(2*np.pi*ci.month/12),
            "month_cos":np.cos(2*np.pi*ci.month/12),
        }])
        cancel_prob = float(m_cancel.predict_proba(inp[CANCEL_FEATS])[0,1])
        los_pred    = int(np.clip(round(m_los.predict(inp[LOS_FEATS])[0]),1,7))
        risk_label  = ("🟢 LOW" if cancel_prob<0.15 else
                       "🟡 MEDIUM" if cancel_prob<0.30 else "🔴 HIGH")
        r1,r2,r3,r4 = st.columns(4)
        cls = "g" if cancel_prob<0.15 else "o" if cancel_prob<0.3 else "r"
        with r1: st.markdown(kpi(f"{cancel_prob*100:.1f}%","Cancel Probability",cls), unsafe_allow_html=True)
        with r2: st.markdown(kpi(risk_label,"Risk Level"), unsafe_allow_html=True)
        with r3: st.markdown(kpi(f"{los_pred} nights","Predicted Stay"), unsafe_allow_html=True)
        with r4: st.markdown(kpi(fmt(input_rate*los_pred),"Expected Revenue"), unsafe_allow_html=True)
        msgs = {"🟢 LOW":"✅ Low risk. Standard confirmation.",
                "🟡 MEDIUM":"⚠️ Moderate risk. Consider deposit or non-refundable upgrade.",
                "🔴 HIGH":"🚨 High risk. Recommend non-refundable rate or overbooking buffer."}
        st.markdown(f'<div class="insight">{msgs[risk_label]}</div>', unsafe_allow_html=True)

    # Historical analysis
    try:
        import altair as alt
        st.markdown('<p class="sec">📊 Historical Cancellation Patterns</p>', unsafe_allow_html=True)
        raw["lead_bucket"] = pd.cut(raw["lead_time"],
            bins=[0,7,30,60,90,365,730],
            labels=["0-7d","8-30d","31-60d","61-90d","91-365d","365d+"],right=True)
        ca1,ca2 = st.columns(2)
        with ca1:
            st.markdown("**Cancel Rate by Lead Time**")
            ltd = raw.groupby("lead_bucket",observed=True)["is_cancelled"].mean().reset_index()
            ltd["pct"] = ltd["is_cancelled"]*100
            st.altair_chart(alt.Chart(ltd).mark_bar(color="#e94560")
                .encode(x=alt.X("lead_bucket:O"),y=alt.Y("pct:Q",title="Cancel %"),
                        tooltip=["lead_bucket:O",alt.Tooltip("pct:Q",format=".1f")])
                .properties(height=240),use_container_width=True)
        with ca2:
            st.markdown("**Cancel Rate by Rate Plan**")
            rpd = raw.groupby("Rate_Plan")["is_cancelled"].mean().reset_index()
            rpd["pct"] = rpd["is_cancelled"]*100
            rpd["Rate_Plan"] = rpd["Rate_Plan"].str.replace(r" \(> 30 days\)","",regex=True)
            st.altair_chart(alt.Chart(rpd).mark_bar(color="#0f3460")
                .encode(x=alt.X("pct:Q",title="Cancel %"),
                        y=alt.Y("Rate_Plan:O",sort="-x"),
                        tooltip=["Rate_Plan:O",alt.Tooltip("pct:Q",format=".1f")])
                .properties(height=240),use_container_width=True)
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LENGTH OF STAY
# ══════════════════════════════════════════════════════════════════════════════
with tab_los:
    st.markdown('<p class="sec">🛏️ Length-of-Stay Analysis & Optimization</p>',
                unsafe_allow_html=True)
    lm = los_metrics
    l1,l2,l3,l4 = st.columns(4)
    with l1: st.markdown(kpi(f"{lm['MAE']:.3f}","MAE (nights)"), unsafe_allow_html=True)
    with l2: st.markdown(kpi(f"{lm['Accuracy']*100:.1f}%","Exact Accuracy","g"), unsafe_allow_html=True)
    with l3: st.markdown(kpi(f"{raw['Number_of_Nights'].mean():.2f}","Avg LOS","o"), unsafe_allow_html=True)
    with l4: st.markdown(kpi(f"{raw['Number_of_Nights'].median():.0f} nts","Median LOS"), unsafe_allow_html=True)

    try:
        import altair as alt
        la1,la2 = st.columns(2)
        with la1:
            st.markdown("**LOS Distribution**")
            vc_act  = pd.Series(lm["test_actual"]).value_counts().sort_index().reset_index()
            vc_pred = pd.Series(lm["test_preds"]).value_counts().sort_index().reset_index()
            vc_act.columns  = ["nights","count"]; vc_act["series"]  = "Actual"
            vc_pred.columns = ["nights","count"]; vc_pred["series"] = "Predicted"
            dist = pd.concat([vc_act,vc_pred])
            st.altair_chart(alt.Chart(dist).mark_bar(opacity=0.7)
                .encode(x=alt.X("nights:O"),y="count:Q",
                        color=alt.Color("series:N",scale=alt.Scale(
                            domain=["Actual","Predicted"],range=["#0f3460","#e94560"])),
                        xOffset="series:N",
                        tooltip=["nights:O","series:N","count:Q"])
                .properties(height=260),use_container_width=True)
        with la2:
            st.markdown("**Avg LOS & Revenue by Nights**")
            conf_r = raw[raw["Cancellation_Status"]=="Confirmed"]
            rv = conf_r.groupby("Number_of_Nights")["Revenue_Generated"].mean().reset_index()
            st.altair_chart(alt.Chart(rv).mark_bar(color="#27ae60")
                .encode(x=alt.X("Number_of_Nights:O"),
                        y=alt.Y("Revenue_Generated:Q",axis=alt.Axis(format="~s")),
                        tooltip=["Number_of_Nights:O",
                                 alt.Tooltip("Revenue_Generated:Q",format=",.0f")])
                .properties(height=260),use_container_width=True)

        st.markdown("**LOS Heatmap — Month × Channel**")
        raw["ci_month"] = raw["Check_in_Date"].dt.strftime("%b")
        los_mc = raw.groupby(["ci_month","Booking_Channel"])["Number_of_Nights"].mean().reset_index()
        st.altair_chart(alt.Chart(los_mc).mark_rect()
            .encode(x=alt.X("Booking_Channel:O"),
                    y=alt.Y("ci_month:O",sort=["Jan","Feb","Mar","Apr","May","Jun",
                                               "Jul","Aug","Sep","Oct","Nov","Dec"]),
                    color=alt.Color("Number_of_Nights:Q",scale=alt.Scale(scheme="blues")),
                    tooltip=["ci_month:O","Booking_Channel:O",
                             alt.Tooltip("Number_of_Nights:Q",format=".2f")])
            .properties(height=300),use_container_width=True)

    except ImportError:
        pass

    # Recommendation
    conf_r2 = raw[raw["Cancellation_Status"]=="Confirmed"]
    best_los = conf_r2.groupby("Number_of_Nights")["Revenue_Generated"].mean().idxmax()
    best_rev = conf_r2.groupby("Number_of_Nights")["Revenue_Generated"].mean()[best_los]
    st.markdown(f"""<div class="insight">
    📌 <b>Highest avg revenue per booking: {best_los}-night stays</b> ({fmt(best_rev)})<br>
    📌 Most common LOS: <b>{raw['Number_of_Nights'].mode()[0]} nights</b>
    ({(raw['Number_of_Nights']==raw['Number_of_Nights'].mode()[0]).mean()*100:.1f}% of bookings)<br>
    📌 <b>Recommendation:</b> Promote {best_los}-night minimum stay packages to maximise revenue per booking.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CHANNEL MIX
# ══════════════════════════════════════════════════════════════════════════════
with tab_channel:
    st.markdown(f'<p class="sec">📡 Channel Mix Forecast — {tomorrow_str} to {end_str}</p>',
                unsafe_allow_html=True)

    conf_exp = expanded[expanded["Cancellation_Status"]=="Confirmed"]
    ch_share = conf_exp["Booking_Channel"].value_counts(normalize=True)*100
    c1,c2,c3,c4 = st.columns(4)
    for col,ch in zip([c1,c2,c3,c4],CHANNELS):
        with col: st.markdown(kpi(f"{ch_share.get(ch,0):.1f}%",f"{ch} Share"),
                               unsafe_allow_html=True)

    try:
        import altair as alt

        st.markdown("**Forecast: Rooms per Channel (Stacked Area)**")
        ch_melt = fcast[["stay_date"]+CHANNELS].melt("stay_date",
                                                       var_name="Channel",value_name="Rooms")
        st.altair_chart(alt.Chart(ch_melt).mark_area(opacity=0.85)
            .encode(x=alt.X("stay_date:T"),y=alt.Y("Rooms:Q",stack="zero"),
                    color=alt.Color("Channel:N",scale=alt.Scale(
                        domain=CHANNELS,range=[CHANNEL_COLORS[c] for c in CHANNELS])),
                    tooltip=["stay_date:T","Channel:N",alt.Tooltip("Rooms:Q",format=".1f")])
            .properties(height=280),use_container_width=True)

        st.markdown("**Historical: Daily Rooms by Channel (last 180 days)**")
        hist_ch = daily.tail(180)[["stay_date"]+CHANNELS].melt(
            "stay_date",var_name="Channel",value_name="Rooms")
        st.altair_chart(alt.Chart(hist_ch).mark_line(strokeWidth=1.5)
            .encode(x="stay_date:T",y="Rooms:Q",
                    color=alt.Color("Channel:N",scale=alt.Scale(
                        domain=CHANNELS,range=[CHANNEL_COLORS[c] for c in CHANNELS])),
                    tooltip=["stay_date:T","Channel:N",alt.Tooltip("Rooms:Q",format=".0f")])
            .properties(height=240),use_container_width=True)

        # Revenue by channel
        st.markdown("**Revenue Contribution by Channel**")
        ch_rev = (conf_exp.groupby("Booking_Channel")["Revenue_Generated"]
                  .sum().reset_index().sort_values("Revenue_Generated",ascending=False))
        ch_rev["pct"] = ch_rev["Revenue_Generated"]/ch_rev["Revenue_Generated"].sum()*100
        st.altair_chart(alt.Chart(ch_rev).mark_bar()
            .encode(x=alt.X("Revenue_Generated:Q",axis=alt.Axis(format="~s")),
                    y=alt.Y("Booking_Channel:O",sort="-x"),
                    color=alt.Color("Booking_Channel:N",scale=alt.Scale(
                        domain=CHANNELS,range=[CHANNEL_COLORS[c] for c in CHANNELS])),
                    tooltip=["Booking_Channel:O",
                             alt.Tooltip("Revenue_Generated:Q",format=",.0f"),
                             alt.Tooltip("pct:Q",format=".1f",title="Share %")])
            .properties(height=200),use_container_width=True)
    except ImportError:
        st.line_chart(daily.set_index("stay_date")[CHANNELS])

    # Insights
    st.markdown('<p class="sec">💡 Channel Insights</p>', unsafe_allow_html=True)
    ci1,ci2,ci3 = st.columns(3)
    ch_adr   = conf_exp.groupby("Booking_Channel")["Booked_Rate"].mean().sort_values(ascending=False)
    ch_canc  = raw.groupby("Booking_Channel")["is_cancelled"].mean()*100
    ch_los   = raw[raw["Cancellation_Status"]=="Confirmed"].groupby("Booking_Channel")["Number_of_Nights"].mean()
    with ci1:
        rows_str = "".join([f"{ch}: <b>{fmt(v)}</b><br>" for ch,v in ch_adr.items()])
        st.markdown(f"""<div class="insight"><b>💰 ADR by Channel</b><br>{rows_str}</div>""",
                    unsafe_allow_html=True)
    with ci2:
        rows_str = "".join([f"{ch}: <b>{v:.1f}%</b><br>" for ch,v in ch_canc.sort_values().items()])
        st.markdown(f"""<div class="insight"><b>🚨 Cancel Rate by Channel</b><br>{rows_str}<br>
        Lowest risk: <b>{ch_canc.idxmin()}</b></div>""", unsafe_allow_html=True)
    with ci3:
        rows_str = "".join([f"{ch}: <b>{v:.2f} nts</b><br>" for ch,v in ch_los.sort_values(ascending=False).items()])
        st.markdown(f"""<div class="insight"><b>🛏️ Avg LOS by Channel</b><br>{rows_str}</div>""",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown('<p class="sec">📊 Model Diagnostics</p>', unsafe_allow_html=True)
    diag_t = st.selectbox("Select model",list(ts_metrics.keys()),
                           format_func=lambda x:LABEL_MAP.get(x,x))
    dm = ts_metrics[diag_t]
    diag_df = pd.DataFrame({
        "date": pd.to_datetime(dm["test_dates"]),
        "Actual": dm["test_actual"],
        "Predicted": dm["test_preds"],
    })
    diag_df["residual"] = diag_df["Actual"] - diag_df["Predicted"]
    diag_df["abs_pct"]  = (np.abs(diag_df["residual"]) /
                            np.where(diag_df["Actual"]==0,1,diag_df["Actual"]))*100
    d1,d2,d3,d4 = st.columns(4)
    with d1: st.markdown(kpi(f"{dm['MAE']:,.1f}","MAE"), unsafe_allow_html=True)
    with d2: st.markdown(kpi(f"{dm['MAPE']:.2f}%","MAPE","g"), unsafe_allow_html=True)
    with d3: st.markdown(kpi(f"{dm['R2']:.4f}","R²","g"), unsafe_allow_html=True)
    with d4: st.markdown(kpi(f"{diag_df['abs_pct'].median():.1f}%","Median Abs Err%"), unsafe_allow_html=True)
    try:
        import altair as alt
        melt = diag_df[["date","Actual","Predicted"]].melt("date")
        st.altair_chart(alt.Chart(melt).mark_line(strokeWidth=1.8)
            .encode(x="date:T",y=alt.Y("value:Q",axis=alt.Axis(format="~s")),
                    color=alt.Color("variable:N",scale=alt.Scale(
                        domain=["Actual","Predicted"],range=["#0f3460","#e94560"])),
                    tooltip=["date:T",alt.Tooltip("value:Q",format=",.1f"),"variable:N"])
            .properties(height=300),use_container_width=True)
        dc1,dc2 = st.columns(2)
        with dc1:
            st.altair_chart(alt.Chart(diag_df).mark_bar(color="#e94560",opacity=.7)
                .encode(x="date:T",y=alt.Y("residual:Q",axis=alt.Axis(format="~s")),
                        tooltip=["date:T",alt.Tooltip("residual:Q",format=",.1f")])
                .properties(height=220,title="Residuals"),use_container_width=True)
        with dc2:
            st.altair_chart(alt.Chart(diag_df).mark_area(color="#0f3460",opacity=.5)
                .encode(x="date:T",y=alt.Y("abs_pct:Q",title="Abs % Error"),
                        tooltip=["date:T",alt.Tooltip("abs_pct:Q",format=".1f")])
                .properties(height=220,title="Abs % Error"),use_container_width=True)
    except ImportError:
        st.line_chart(diag_df.set_index("date")[["Actual","Predicted"]])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_fi:
    st.markdown('<p class="sec">🏷️ Feature Importance</p>', unsafe_allow_html=True)
    fi_src = st.radio("Model type",["Time-Series","Booking-Level"],horizontal=True)
    if fi_src == "Time-Series":
        fi_t = st.selectbox("Model",list(ts_models.keys()),
                             format_func=lambda x:LABEL_MAP.get(x,x))
        fi_df = pd.DataFrame({"Feature":feat_cols,
                               "Importance":ts_models[fi_t].feature_importances_}).sort_values(
            "Importance",ascending=False).head(25)
    else:
        fi_bk = st.selectbox("Model",["Cancellation","LOS"])
        src   = cancel_metrics if fi_bk=="Cancellation" else los_metrics
        fi_df = pd.DataFrame({"Feature":src["features"],
                               "Importance":src["importances"]}).sort_values(
            "Importance",ascending=False).head(20)
    try:
        import altair as alt
        st.altair_chart(alt.Chart(fi_df).mark_bar(color="#0f3460")
            .encode(x="Importance:Q",y=alt.Y("Feature:N",sort="-x"),
                    tooltip=["Feature:N","Importance:Q"])
            .properties(height=560),use_container_width=True)
    except ImportError:
        st.bar_chart(fi_df.set_index("Feature")["Importance"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<p class="sec">📋 OTB Snapshot & Daily Master</p>', unsafe_allow_html=True)

    st.markdown(f"**On-the-Book as of {as_of_date} (next {horizon} days)**")
    otb_show = otb_df.copy()
    otb_show["stay_date"]   = otb_show["stay_date"].dt.strftime("%a %d %b")
    otb_show["occ_pct_otb"] = otb_show["occ_pct_otb"].apply(lambda x:f"{x:.1f}%")
    otb_show["revenue_otb"] = otb_show["revenue_otb"].apply(lambda x:f"{x:,.0f}")
    st.dataframe(otb_show[["stay_date","rooms_otb","occ_pct_otb","revenue_otb",
                             "std_otb","dlx_otb","ste_otb","remaining_rooms"]],
                 use_container_width=True, height=300)

    st.markdown("**Historical Daily Master**")
    view = daily[["stay_date","nightly_revenue","rooms_occupied","occ_pct_total",
                   "occ_pct_Standard","occ_pct_Deluxe","occ_pct_Suite","adr","revpar"]+CHANNELS].copy()
    view["stay_date"] = view["stay_date"].dt.strftime("%Y-%m-%d")
    st.dataframe(view, use_container_width=True, height=350)

    dl = daily.copy(); dl["stay_date"] = dl["stay_date"].astype(str)
    st.download_button("⬇️ Download Daily Master CSV",
                        dl.to_csv(index=False).encode(),"daily_master.csv","text/csv")
    dl2 = fcast.copy(); dl2["stay_date"] = dl2["stay_date"].astype(str)
    st.download_button("⬇️ Download OTB Forecast CSV",
                        dl2.to_csv(index=False).encode(),
                        f"otb_forecast_{as_of_date}_{horizon}d.csv","text/csv")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#aaa;font-size:.75rem'>"
    "Hotel Revenue Intelligence • OTB-Anchored • 12 LightGBM Models • "
    "Revenue | Occupancy | ADR | Cancellation | LOS | Channel Mix</center>",
    unsafe_allow_html=True,
)
