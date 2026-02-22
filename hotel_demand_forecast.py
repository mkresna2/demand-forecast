"""
Hotel Revenue Intelligence Platform
────────────────────────────────────
Models:
  Time-series (stay_date level):
    1. Nightly Revenue         LightGBM Regressor
    2. Occupancy % Overall     LightGBM Regressor
    3. Occupancy % Standard    LightGBM Regressor
    4. Occupancy % Deluxe      LightGBM Regressor
    5. Occupancy % Suite       LightGBM Regressor
    6. ADR (Avg Daily Rate)    LightGBM Regressor
    7-10. Channel Mix          LightGBM Regressor × 4

  Booking-level:
    11. Cancellation Prob      LightGBM Classifier
    12. Length-of-Stay         LightGBM Regressor
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
  .hero {
    background: linear-gradient(135deg,#0d0d1a 0%,#1a1a2e 40%,#0f3460 100%);
    padding:2rem 2.5rem; border-radius:14px; margin-bottom:1.5rem; color:#fff;
  }
  .hero h1 { font-size:2rem; font-weight:800; margin:0; letter-spacing:-.5px; }
  .hero p  { opacity:.7; margin:.4rem 0 0; font-size:.95rem; }
  .kpi {
    background:#fff; border-radius:10px; padding:1rem;
    box-shadow:0 2px 10px rgba(0,0,0,.07); text-align:center;
    border-left:4px solid #0f3460; height:100%;
  }
  .kpi .v { font-size:1.6rem; font-weight:700; color:#0f3460; margin:0; line-height:1.2; }
  .kpi .l { font-size:.75rem; color:#999; margin:.25rem 0 0; }
  .kpi.g .v { color:#27ae60; } .kpi.g { border-left-color:#27ae60; }
  .kpi.o .v { color:#e67e22; } .kpi.o { border-left-color:#e67e22; }
  .kpi.r .v { color:#e74c3c; } .kpi.r { border-left-color:#e74c3c; }
  .kpi.p .v { color:#8e44ad; } .kpi.p { border-left-color:#8e44ad; }
  .insight { background:#f8f9ff; border-left:3px solid #0f3460;
             border-radius:8px; padding:.85rem 1rem; margin:.4rem 0;
             font-size:.88rem; line-height:1.6; }
  .warn    { background:#fff8f0; border-left:3px solid #e67e22;
             border-radius:8px; padding:.85rem 1rem; margin:.4rem 0;
             font-size:.88rem; }
  .sec { font-size:1.1rem; font-weight:700; color:#1a1a2e;
         margin:1.4rem 0 .7rem; padding-bottom:.3rem;
         border-bottom:2px solid #e8ecf4; }
  .tag { display:inline-block; padding:.2rem .6rem; border-radius:20px;
         font-size:.75rem; font-weight:600; margin:.1rem; }
  .tag-blue  { background:#e8f0fe; color:#1a73e8; }
  .tag-green { background:#e6f4ea; color:#1e8e3e; }
  .tag-red   { background:#fce8e6; color:#d93025; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CAPACITY    = {"Standard": 50, "Deluxe": 25, "Suite": 10, "total": 85}
ROOM_COLORS = {"Standard": "#0f3460", "Deluxe": "#e94560", "Suite": "#e67e22"}
CHANNEL_COLORS = {"Direct": "#0f3460", "OTA": "#e94560", "Walk-in": "#27ae60", "Website": "#8e44ad"}
CHANNELS    = ["Direct", "OTA", "Walk-in", "Website"]

TS_TARGETS  = ["nightly_revenue", "occ_pct_total",
               "occ_pct_Standard", "occ_pct_Deluxe", "occ_pct_Suite", "adr"]
LABEL_MAP   = {
    "nightly_revenue":  "Nightly Revenue",
    "occ_pct_total":    "Occupancy % (Overall)",
    "occ_pct_Standard": "Occupancy % (Standard)",
    "occ_pct_Deluxe":   "Occupancy % (Deluxe)",
    "occ_pct_Suite":    "Occupancy % (Suite)",
    "adr":              "ADR (Avg Daily Rate)",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt(val, prefix="Rp "):
    if   val >= 1e9: return f"{prefix}{val/1e9:.2f}B"
    elif val >= 1e6: return f"{prefix}{val/1e6:.1f}M"
    elif val >= 1e3: return f"{prefix}{val/1e3:.1f}K"
    return f"{prefix}{val:,.0f}"

def kpi(val, label, cls=""):
    return (f'<div class="kpi {cls}">'
            f'<p class="v">{val}</p><p class="l">{label}</p></div>')

def altair_line(df, x, ys, colors, title="", height=280, y_fmt="~s", y_title=""):
    try:
        import altair as alt
        melt = df[[x]+ys].melt(x, var_name="Series", value_name="value")
        return (alt.Chart(melt)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X(f"{x}:T", title="Date"),
                    y=alt.Y("value:Q", title=y_title,
                             axis=alt.Axis(format=y_fmt)),
                    color=alt.Color("Series:N",
                        scale=alt.Scale(domain=ys, range=colors)),
                    strokeDash=alt.condition(
                        alt.datum.Series.endswith("(F)"),
                        alt.value([5,3]), alt.value([0])),
                    tooltip=[f"{x}:T",
                             alt.Tooltip("value:Q", format=",.1f"), "Series:N"],
                ).properties(title=title, height=height))
    except ImportError:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_expanded(raw: pd.DataFrame):
    """Expand each booking to one row per stay night."""
    raw = raw.copy()
    raw["Check_in_Date"]  = pd.to_datetime(raw["Check_in_Date"])
    raw["Check_out_Date"] = pd.to_datetime(raw["Check_out_Date"])
    raw["Booking_Date"]   = pd.to_datetime(raw["Booking_Date"])
    raw["is_cancelled"]   = (raw["Cancellation_Status"] == "Cancelled").astype(int)
    raw["lead_time"]      = (raw["Check_in_Date"] - raw["Booking_Date"]).dt.days

    rows = []
    for _, r in raw.iterrows():
        for d in pd.date_range(r["Check_in_Date"],
                                r["Check_out_Date"] - timedelta(days=1)):
            rec = r.to_dict()
            rec["stay_date"] = d
            rows.append(rec)

    expanded = pd.DataFrame(rows)
    return raw, expanded


@st.cache_data(show_spinner=False)
def build_daily(expanded: pd.DataFrame) -> pd.DataFrame:
    """Aggregate confirmed stay-night rows to daily master table."""
    conf = expanded[expanded["Cancellation_Status"] == "Confirmed"].copy()

    base = conf.groupby("stay_date").agg(
        rooms_occupied    = ("Booking_ID", "count"),
        nightly_revenue   = ("Booked_Rate", "sum"),
        rooms_Standard    = ("Room_Type", lambda x: (x == "Standard").sum()),
        rooms_Deluxe      = ("Room_Type", lambda x: (x == "Deluxe").sum()),
        rooms_Suite       = ("Room_Type", lambda x: (x == "Suite").sum()),
        avg_lead_time     = ("lead_time", "mean"),
        avg_nights        = ("Number_of_Nights", "mean"),
        avg_guests        = ("Number_of_Guests", "mean"),
    ).reset_index()

    # Channel pivot
    ch_piv = (conf.groupby(["stay_date", "Booking_Channel"])["Booking_ID"]
              .count().unstack(fill_value=0).reset_index())
    ch_piv.columns.name = None
    for ch in CHANNELS:
        if ch not in ch_piv.columns:
            ch_piv[ch] = 0
    base = base.merge(ch_piv[["stay_date"] + CHANNELS], on="stay_date", how="left").fillna(0)

    # Occupancy %
    for rt in ["Standard", "Deluxe", "Suite"]:
        base[f"occ_pct_{rt}"] = base[f"rooms_{rt}"] / CAPACITY[rt] * 100
    base["occ_pct_total"] = base["rooms_occupied"] / CAPACITY["total"] * 100

    # ADR: average Booked_Rate across confirmed stay-nights
    adr = conf.groupby("stay_date")["Booked_Rate"].mean().reset_index()
    adr.columns = ["stay_date", "adr"]
    base = base.merge(adr, on="stay_date", how="left")

    # RevPAR
    base["revpar"] = base["nightly_revenue"] / CAPACITY["total"]

    # Channel mix %
    ch_total = base[CHANNELS].sum(axis=1).replace(0, 1)
    for ch in CHANNELS:
        base[f"pct_{ch}"] = base[ch] / ch_total * 100

    return base.sort_values("stay_date").reset_index(drop=True)


def _add_time(df):
    df = df.copy()
    df["dow"]        = df["stay_date"].dt.dayofweek
    df["month"]      = df["stay_date"].dt.month
    df["week"]       = df["stay_date"].dt.isocalendar().week.astype(int)
    df["quarter"]    = df["stay_date"].dt.quarter
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["year"]       = df["stay_date"].dt.year
    df["dom"]        = df["stay_date"].dt.day
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def _add_lags(df, targets):
    df = df.copy()
    for t in targets:
        if t not in df.columns:
            continue
        for lag in [1, 2, 3, 7, 14, 21, 28, 30, 60, 90]:
            df[f"{t}_lag{lag}"] = df[t].shift(lag)
        for w in [3, 7, 14, 30, 60]:
            df[f"{t}_roll{w}"]    = df[t].shift(1).rolling(w).mean()
            df[f"{t}_roll{w}std"] = df[t].shift(1).rolling(w).std()
    return df


def _feat_cols(df, drop_extra=None):
    drop = {"stay_date"} | set(drop_extra or [])
    return [c for c in df.columns if c not in drop]


# ══════════════════════════════════════════════════════════════════════════════
# TIME-SERIES MODELS  (nightly_revenue, occ_pct_*, adr, channel mix)
# ══════════════════════════════════════════════════════════════════════════════

def train_ts_models(daily: pd.DataFrame):
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error

    all_ts_targets = TS_TARGETS + CHANNELS
    df = _add_time(daily)
    df = _add_lags(df, all_ts_targets)
    df = df.dropna().reset_index(drop=True)

    non_feat = ({"stay_date"} | set(all_ts_targets) |
                {"rooms_occupied","rooms_Standard","rooms_Deluxe","rooms_Suite",
                 "revpar"} |
                {f"pct_{ch}" for ch in CHANNELS})
    feat_cols = [c for c in df.columns if c not in non_feat]

    split = len(df) - 90
    models, metrics = {}, {}

    for target in all_ts_targets:
        if target not in df.columns:
            continue
        X = df[feat_cols]
        y = df[target]
        Xtr, Xte = X.iloc[:split], X.iloc[split:]
        ytr, yte  = y.iloc[:split], y.iloc[split:]

        m = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.02, max_depth=7,
            num_leaves=63, min_child_samples=5, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.05,
            random_state=42, verbose=-1,
        )
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

        preds = np.clip(m.predict(Xte), 0, None)
        if "occ_pct" in target:
            preds = np.minimum(preds, 100.0)

        mae  = mean_absolute_error(yte, preds)
        denom = np.where(yte.values == 0, 1, yte.values)
        mape = float(np.mean(np.abs((yte.values - preds) / denom)) * 100)
        r2   = float(1 - np.sum((yte.values-preds)**2) /
                     np.sum((yte.values - yte.mean())**2))

        models[target]  = m
        metrics[target] = {
            "MAE": mae, "MAPE": mape, "R2": r2,
            "test_preds":  preds, "test_actual": yte.values,
            "test_dates":  df["stay_date"].iloc[split:].values,
        }

    return models, metrics, feat_cols, df


# ══════════════════════════════════════════════════════════════════════════════
# BOOKING-LEVEL MODELS  (cancellation + LOS)
# ══════════════════════════════════════════════════════════════════════════════

def _booking_features(raw: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    df = raw.copy()
    df["lead_time"]      = (df["Check_in_Date"] - df["Booking_Date"]).dt.days
    df["checkin_dow"]    = df["Check_in_Date"].dt.dayofweek
    df["checkin_month"]  = df["Check_in_Date"].dt.month
    df["checkin_quarter"]= df["Check_in_Date"].dt.quarter
    df["checkin_week"]   = df["Check_in_Date"].dt.isocalendar().week.astype(int)
    df["booking_dow"]    = df["Booking_Date"].dt.dayofweek
    df["booking_month"]  = df["Booking_Date"].dt.month
    df["booking_year"]   = df["Booking_Date"].dt.year
    df["is_member"]      = df["Rate_Plan"].str.contains("Member").astype(int)
    df["is_early_bird"]  = df["Rate_Plan"].str.contains("Early Bird").astype(int)
    df["is_non_refund"]  = df["Rate_Plan"].str.contains("Non-Refundable").astype(int)
    df["is_bar"]         = df["Rate_Plan"].str.contains("BAR").astype(int)
    df["is_corporate"]   = df["Rate_Plan"].str.contains("Corporate").astype(int)
    df["is_weekend_ci"]  = (df["checkin_dow"] >= 5).astype(int)
    df["lead_time_sq"]   = df["lead_time"] ** 2
    df["nights_x_guests"]= df["Number_of_Nights"] * df["Number_of_Guests"]
    df["rev_per_guest"]  = df["Booked_Rate"] / df["Number_of_Guests"]
    df["rev_per_night_guest"] = df["Booked_Rate"] / (
        df["Number_of_Guests"] * df["Number_of_Nights"])
    df["month_sin"]      = np.sin(2 * np.pi * df["checkin_month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["checkin_month"] / 12)

    for col in ["Room_Type", "Booking_Channel", "Rate_Plan"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col])

    return df


CANCEL_FEATS = [
    "lead_time", "lead_time_sq", "Number_of_Nights", "Number_of_Guests",
    "nights_x_guests", "Room_Type_enc", "Booking_Channel_enc", "Rate_Plan_enc",
    "checkin_dow", "checkin_month", "checkin_quarter", "checkin_week",
    "booking_dow", "booking_month", "booking_year",
    "is_member", "is_early_bird", "is_non_refund", "is_bar", "is_corporate",
    "is_weekend_ci", "Booked_Rate", "rev_per_guest", "rev_per_night_guest",
    "month_sin", "month_cos",
]
LOS_FEATS = [f for f in CANCEL_FEATS
             if f not in ("Number_of_Nights", "nights_x_guests", "rev_per_night_guest")]


def train_booking_models(raw: pd.DataFrame):
    import lightgbm as lgb
    from sklearn.metrics import (mean_absolute_error, roc_auc_score,
                                  accuracy_score, classification_report)
    df = _booking_features(raw)
    split = int(len(df) * 0.8)

    # ── Cancellation ──────────────────────────────────────────────────────────
    Xtr = df[CANCEL_FEATS].iloc[:split]; Xte = df[CANCEL_FEATS].iloc[split:]
    ytr = df["is_cancelled"].iloc[:split]; yte = df["is_cancelled"].iloc[split:]

    m_cancel = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, num_leaves=63,
        class_weight="balanced", subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    m_cancel.fit(Xtr, ytr, eval_set=[(Xte, yte)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    proba    = m_cancel.predict_proba(Xte)[:, 1]
    preds_c  = (proba > 0.5).astype(int)
    auc      = float(roc_auc_score(yte, proba))
    acc      = float(accuracy_score(yte, preds_c))
    cancel_report = classification_report(yte, preds_c, output_dict=True)

    cancel_metrics = {
        "AUC": auc, "Accuracy": acc,
        "report": cancel_report,
        "test_proba":  proba,
        "test_actual": yte.values,
        "features": CANCEL_FEATS,
        "importances": m_cancel.feature_importances_,
    }

    # ── Length of Stay ─────────────────────────────────────────────────────────
    Xtr_l = df[LOS_FEATS].iloc[:split]; Xte_l = df[LOS_FEATS].iloc[split:]
    ytr_l = df["Number_of_Nights"].iloc[:split]; yte_l = df["Number_of_Nights"].iloc[split:]

    m_los = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        random_state=42, verbose=-1,
    )
    m_los.fit(Xtr_l, ytr_l)
    preds_los = np.clip(np.round(m_los.predict(Xte_l)), 1, 7)
    mae_los   = float(mean_absolute_error(yte_l, preds_los))
    acc_los   = float((preds_los == yte_l.values).mean())

    los_metrics = {
        "MAE": mae_los, "Accuracy": acc_los,
        "test_preds":  preds_los,
        "test_actual": yte_l.values,
        "features": LOS_FEATS,
        "importances": m_los.feature_importances_,
        "raw_df": df,   # keep for input widget
    }

    return m_cancel, cancel_metrics, m_los, los_metrics


# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

def forecast_ts(ts_models, feat_cols, daily, horizon=30):
    """Iteratively forecast all time-series targets."""
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
    for i in range(1, horizon + 1):
        next_date = pd.Timestamp(daily["stay_date"].max() + timedelta(days=i))
        new_row   = {"stay_date": next_date}

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

        pred = {}
        for t, m in ts_models.items():
            v = float(np.clip(m.predict(row_f)[0], 0, None))
            if "occ_pct" in t:
                v = min(v, 100.0)
            pred[t] = v
            history.at[history.index[-1], t] = v

        # Derived
        history.at[history.index[-1], "revpar"] = (
            pred.get("nightly_revenue", 0) / CAPACITY["total"])
        history.at[history.index[-1], "rooms_occupied"] = (
            pred.get("occ_pct_total", 0) * CAPACITY["total"] / 100)

        records.append({"stay_date": next_date, **pred,
                         "revpar": pred.get("nightly_revenue", 0) / CAPACITY["total"],
                         "rooms_occupied": pred.get("occ_pct_total",0)*CAPACITY["total"]/100})

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
  <p>LightGBM • Revenue • Occupancy • ADR • Cancellation Risk •
     Length-of-Stay • Channel Mix</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    uploaded = st.file_uploader("📂 Upload Bookings CSV", type=["csv"])
    st.markdown("---")

    st.markdown("### 🔮 Forecast")
    horizon    = st.slider("Horizon (days)", 7, 180, 30, 7)
    show_by_rt = st.toggle("Occupancy by Room Type", value=True)

    with st.expander("🏗️ Room Capacity"):
        CAPACITY["Standard"] = st.number_input("Standard", 1, 500, 50)
        CAPACITY["Deluxe"]   = st.number_input("Deluxe",   1, 500, 25)
        CAPACITY["Suite"]    = st.number_input("Suite",    1, 500, 10)
        CAPACITY["total"]    = CAPACITY["Standard"] + CAPACITY["Deluxe"] + CAPACITY["Suite"]

    st.markdown("---")
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        if uploaded is None:
            st.error("Upload a CSV first.")
        else:
            prog = st.progress(0, "Loading data…")
            raw_df = pd.read_csv(uploaded)
            raw_df["Check_in_Date"]  = pd.to_datetime(raw_df["Check_in_Date"])
            raw_df["Check_out_Date"] = pd.to_datetime(raw_df["Check_out_Date"])
            raw_df["Booking_Date"]   = pd.to_datetime(raw_df["Booking_Date"])

            prog.progress(15, "Expanding stay nights…")
            raw_proc, expanded = build_expanded(raw_df)

            prog.progress(30, "Building daily aggregates…")
            daily = build_daily(expanded)

            prog.progress(45, "Training time-series models…")
            ts_models, ts_metrics, feat_cols, ts_df = train_ts_models(daily)

            prog.progress(80, "Training booking-level models…")
            m_cancel, cancel_met, m_los, los_met = train_booking_models(raw_proc)

            prog.progress(100, "Done!")

            st.session_state.ts_models      = ts_models
            st.session_state.ts_metrics     = ts_metrics
            st.session_state.ts_feat_cols   = feat_cols
            st.session_state.ts_df          = ts_df
            st.session_state.daily          = daily
            st.session_state.m_cancel       = m_cancel
            st.session_state.cancel_metrics = cancel_met
            st.session_state.m_los          = m_los
            st.session_state.los_metrics    = los_met
            st.session_state.raw            = raw_proc
            st.session_state.expanded       = expanded
            prog.empty()
            st.success("✅ All models trained!")

    # Model summary
    if st.session_state.ts_models:
        st.markdown("### 📊 Model Summary")
        for t in TS_TARGETS:
            if t in st.session_state.ts_metrics:
                m = st.session_state.ts_metrics[t]
                lbl = LABEL_MAP.get(t, t)
                st.caption(f"**{lbl}** R²={m['R2']:.3f} MAPE={m['MAPE']:.1f}%")
        if st.session_state.cancel_metrics:
            cm = st.session_state.cancel_metrics
            st.caption(f"**Cancellation** AUC={cm['AUC']:.3f} Acc={cm['Accuracy']:.3f}")
        if st.session_state.los_metrics:
            lm = st.session_state.los_metrics
            st.caption(f"**LOS** MAE={lm['MAE']:.2f}nts Acc={lm['Accuracy']:.3f}")

    st.markdown("---")
    st.caption("12 LightGBM models | Stay-date expansion | 137+ features")


# ══════════════════════════════════════════════════════════════════════════════
# GUARD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.ts_models is None:
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, s, d in [
        (c1,"📂","1","Upload Bookings CSV"),
        (c2,"🤖","2","Click Train All Models"),
        (c3,"🔮","3","Explore Demand Forecasts"),
        (c4,"🎯","4","Score Individual Bookings"),
    ]:
        with col:
            st.markdown(kpi(icon, f"Step {s}: {d}"), unsafe_allow_html=True)
    st.info("👈 Upload **Bookings.csv** and click **Train All Models** to begin.")
    st.stop()

# ── Unpack ─────────────────────────────────────────────────────────────────────
ts_models      = st.session_state.ts_models
ts_metrics     = st.session_state.ts_metrics
feat_cols      = st.session_state.ts_feat_cols
daily          = st.session_state.daily
m_cancel       = st.session_state.m_cancel
cancel_metrics = st.session_state.cancel_metrics
m_los          = st.session_state.m_los
los_metrics    = st.session_state.los_metrics
raw            = st.session_state.raw

# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="sec">📈 Model Performance Overview</p>', unsafe_allow_html=True)
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
rm = ts_metrics.get("nightly_revenue",{})
om = ts_metrics.get("occ_pct_total",{})
am = ts_metrics.get("adr",{})
with c1: st.markdown(kpi(f"{rm.get('R2',0):.3f}", "Revenue R²","g"), unsafe_allow_html=True)
with c2: st.markdown(kpi(f"{rm.get('MAPE',0):.1f}%","Revenue MAPE"), unsafe_allow_html=True)
with c3: st.markdown(kpi(f"{om.get('R2',0):.3f}", "Occ% R²","o"), unsafe_allow_html=True)
with c4: st.markdown(kpi(f"{am.get('MAPE',0):.2f}%","ADR MAPE","p"), unsafe_allow_html=True)
with c5: st.markdown(kpi(f"{cancel_metrics['AUC']:.3f}","Cancel AUC","r"), unsafe_allow_html=True)
with c6: st.markdown(kpi(f"{los_metrics['MAE']:.2f}nts","LOS MAE"), unsafe_allow_html=True)
with c7: st.markdown(kpi(f"{ts_metrics.get('OTA',{}).get('R2',0):.3f}","Channel R²","g"), unsafe_allow_html=True)

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
# TAB 1 — DEMAND FORECAST  (Revenue + Occupancy)
# ══════════════════════════════════════════════════════════════════════════════
with tab_fcast:
    st.markdown(f'<p class="sec">🔮 {horizon}-Day Revenue & Occupancy Forecast</p>',
                unsafe_allow_html=True)
    with st.spinner("Generating forecast…"):
        fcast = forecast_ts(ts_models, feat_cols, daily, horizon)

    # KPIs
    total_rev = fcast["nightly_revenue"].sum()
    avg_occ   = fcast["occ_pct_total"].mean()
    avg_rpar  = fcast["revpar"].mean()
    peak_rev  = fcast.loc[fcast["nightly_revenue"].idxmax()]
    peak_occ  = fcast.loc[fcast["occ_pct_total"].idxmax()]

    m1,m2,m3,m4,m5 = st.columns(5)
    with m1: st.metric("Total Revenue", fmt(total_rev))
    with m2: st.metric("Avg Occupancy", f"{avg_occ:.1f}%")
    with m3: st.metric("Avg RevPAR", fmt(avg_rpar))
    with m4: st.metric("Peak Revenue", peak_rev["stay_date"].strftime("%d %b"),
                        fmt(peak_rev["nightly_revenue"]))
    with m5: st.metric("Peak Occupancy", peak_occ["stay_date"].strftime("%d %b"),
                        f"{peak_occ['occ_pct_total']:.1f}%")

    try:
        import altair as alt

        # Revenue: hist + forecast
        h_rev = daily[["stay_date","nightly_revenue"]].tail(90).copy()
        h_rev["series"] = "Historical"
        f_rev = fcast[["stay_date","nightly_revenue"]].copy()
        f_rev["series"] = "Forecast"
        rev_df = pd.concat([h_rev.rename(columns={"nightly_revenue":"value"}),
                             f_rev.rename(columns={"nightly_revenue":"value"})])
        rc = (alt.Chart(rev_df).mark_line(strokeWidth=2)
              .encode(x=alt.X("stay_date:T",title="Date"),
                      y=alt.Y("value:Q",axis=alt.Axis(format="~s"),title="Revenue (IDR)"),
                      color=alt.Color("series:N",scale=alt.Scale(
                          domain=["Historical","Forecast"],range=["#0f3460","#e94560"])),
                      strokeDash=alt.condition(
                          alt.datum.series=="Forecast",alt.value([5,3]),alt.value([0])),
                      tooltip=["stay_date:T",alt.Tooltip("value:Q",format=",.0f"),"series:N"])
              .properties(title="Nightly Revenue — Historical (90d) + Forecast",height=280))
        st.altair_chart(rc, use_container_width=True)

        # Occupancy: hist + forecast
        h_occ = daily[["stay_date","occ_pct_total"]].tail(90).copy()
        h_occ["series"] = "Historical"
        f_occ = fcast[["stay_date","occ_pct_total"]].copy()
        f_occ["series"] = "Forecast"
        occ_df = pd.concat([h_occ.rename(columns={"occ_pct_total":"value"}),
                             f_occ.rename(columns={"occ_pct_total":"value"})])
        oc = (alt.Chart(occ_df).mark_line(strokeWidth=2)
              .encode(x=alt.X("stay_date:T",title="Date"),
                      y=alt.Y("value:Q",scale=alt.Scale(domain=[0,100]),title="Occupancy %"),
                      color=alt.Color("series:N",scale=alt.Scale(
                          domain=["Historical","Forecast"],range=["#0f3460","#27ae60"])),
                      strokeDash=alt.condition(
                          alt.datum.series=="Forecast",alt.value([5,3]),alt.value([0])),
                      tooltip=["stay_date:T",alt.Tooltip("value:Q",format=".1f"),"series:N"])
              .properties(title="Overall Occupancy % — Historical (90d) + Forecast",height=260))
        st.altair_chart(oc, use_container_width=True)

        # By room type
        if show_by_rt:
            st.markdown('<p class="sec">🏷️ Occupancy by Room Type (Forecast)</p>',
                        unsafe_allow_html=True)
            rt_melt = fcast[["stay_date","occ_pct_Standard","occ_pct_Deluxe","occ_pct_Suite"]].melt(
                "stay_date", var_name="Room", value_name="Occ %")
            rt_melt["Room"] = rt_melt["Room"].str.replace("occ_pct_","")
            rtc = (alt.Chart(rt_melt).mark_line(strokeWidth=2,point=True)
                   .encode(x=alt.X("stay_date:T"),
                           y=alt.Y("Occ %:Q",scale=alt.Scale(domain=[0,100])),
                           color=alt.Color("Room:N",scale=alt.Scale(
                               domain=["Standard","Deluxe","Suite"],
                               range=[ROOM_COLORS["Standard"],ROOM_COLORS["Deluxe"],ROOM_COLORS["Suite"]])),
                           tooltip=["stay_date:T","Room:N",alt.Tooltip("Occ %:Q",format=".1f")])
                   .properties(height=270))
            st.altair_chart(rtc, use_container_width=True)

    except ImportError:
        st.line_chart(fcast.set_index("stay_date")[["nightly_revenue","occ_pct_total"]])

    # Insights
    st.markdown('<p class="sec">💡 Insights</p>', unsafe_allow_html=True)
    i1,i2,i3 = st.columns(3)
    wk = fcast["stay_date"].dt.dayofweek >= 5
    with i1:
        st.markdown(f"""<div class="insight"><b>📅 Weekend vs Weekday</b><br>
        Occ: <b>{fcast[wk]['occ_pct_total'].mean():.1f}%</b> vs
             <b>{fcast[~wk]['occ_pct_total'].mean():.1f}%</b><br>
        Revenue: <b>{fmt(fcast[wk]['nightly_revenue'].mean())}</b> vs
                 <b>{fmt(fcast[~wk]['nightly_revenue'].mean())}</b>
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown(f"""<div class="insight"><b>🛏️ Avg Forecast Occupancy</b><br>
        Standard: <b>{fcast['occ_pct_Standard'].mean():.1f}%</b><br>
        Deluxe:   <b>{fcast['occ_pct_Deluxe'].mean():.1f}%</b><br>
        Suite:    <b>{fcast['occ_pct_Suite'].mean():.1f}%</b></div>""", unsafe_allow_html=True)
    with i3:
        mo = fcast.groupby(fcast["stay_date"].dt.to_period("M"))["nightly_revenue"].sum()
        bm = mo.idxmax()
        st.markdown(f"""<div class="insight"><b>📆 Best Forecast Month</b><br>
        <b>{bm}</b> — {fmt(mo[bm])}<br>
        Avg Occ: <b>{fcast[fcast['stay_date'].dt.to_period('M')==bm]['occ_pct_total'].mean():.1f}%</b>
        </div>""", unsafe_allow_html=True)

    dl = fcast.copy(); dl["stay_date"] = dl["stay_date"].astype(str)
    st.download_button("⬇️ Download Forecast CSV",
                        dl.to_csv(index=False).encode(),
                        f"demand_forecast_{horizon}d.csv","text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ADR FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_adr:
    st.markdown(f'<p class="sec">💰 ADR (Average Daily Rate) — {horizon}-Day Forecast</p>',
                unsafe_allow_html=True)

    am = ts_metrics.get("adr", {})
    a1,a2,a3,a4 = st.columns(4)
    with a1: st.markdown(kpi(fmt(am.get("MAE",0)), "MAE"), unsafe_allow_html=True)
    with a2: st.markdown(kpi(f"{am.get('MAPE',0):.2f}%", "MAPE", "g"), unsafe_allow_html=True)
    with a3: st.markdown(kpi(f"{am.get('R2',0):.4f}", "R²", "g"), unsafe_allow_html=True)
    with a4: st.markdown(kpi(fmt(daily["adr"].mean()), "Hist Avg ADR"), unsafe_allow_html=True)

    try:
        import altair as alt

        h_adr = daily[["stay_date","adr"]].tail(90).copy(); h_adr["series"]="Historical"
        f_adr = fcast[["stay_date","adr"]].copy(); f_adr["series"]="Forecast"
        adr_df = pd.concat([h_adr.rename(columns={"adr":"value"}),
                             f_adr.rename(columns={"adr":"value"})])
        ac = (alt.Chart(adr_df).mark_line(strokeWidth=2)
              .encode(x=alt.X("stay_date:T",title="Date"),
                      y=alt.Y("value:Q",axis=alt.Axis(format="~s"),title="ADR (IDR)"),
                      color=alt.Color("series:N",scale=alt.Scale(
                          domain=["Historical","Forecast"],range=["#8e44ad","#e67e22"])),
                      strokeDash=alt.condition(
                          alt.datum.series=="Forecast",alt.value([5,3]),alt.value([0])),
                      tooltip=["stay_date:T",alt.Tooltip("value:Q",format=",.0f"),"series:N"])
              .properties(title="ADR — Historical (90d) + Forecast",height=300))
        st.altair_chart(ac, use_container_width=True)

        # ADR by room type (historical)
        st.markdown('<p class="sec">📊 Historical ADR by Room Type</p>', unsafe_allow_html=True)
        adr_rt = (st.session_state.expanded[st.session_state.expanded["Cancellation_Status"]=="Confirmed"]
                  .groupby(["stay_date","Room_Type"])["Booked_Rate"].mean().reset_index())
        arc = (alt.Chart(adr_rt).mark_line(strokeWidth=1.5, opacity=0.85)
               .encode(x=alt.X("stay_date:T"),
                       y=alt.Y("Booked_Rate:Q",axis=alt.Axis(format="~s"),title="ADR (IDR)"),
                       color=alt.Color("Room_Type:N",scale=alt.Scale(
                           domain=["Standard","Deluxe","Suite"],
                           range=[ROOM_COLORS["Standard"],ROOM_COLORS["Deluxe"],ROOM_COLORS["Suite"]])),
                       tooltip=["stay_date:T","Room_Type:N",
                                alt.Tooltip("Booked_Rate:Q",format=",.0f")])
               .properties(height=260))
        st.altair_chart(arc, use_container_width=True)

        # ADR heatmap by month & dow
        st.markdown('<p class="sec">🗓️ ADR Heatmap — Month × Day of Week</p>',
                    unsafe_allow_html=True)
        heat = daily.copy()
        heat["month_name"] = heat["stay_date"].dt.strftime("%b")
        heat["dow_name"]   = heat["stay_date"].dt.strftime("%a")
        heat_agg = heat.groupby(["month_name","dow_name"])["adr"].mean().reset_index()
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        dow_order   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        hm = (alt.Chart(heat_agg).mark_rect()
              .encode(x=alt.X("dow_name:O",sort=dow_order,title="Day"),
                      y=alt.Y("month_name:O",sort=month_order,title="Month"),
                      color=alt.Color("adr:Q",
                          scale=alt.Scale(scheme="viridis"),title="ADR"),
                      tooltip=["month_name:O","dow_name:O",
                               alt.Tooltip("adr:Q",format=",.0f")])
              .properties(height=300))
        st.altair_chart(hm, use_container_width=True)

    except ImportError:
        st.line_chart(daily.set_index("stay_date")["adr"])

    st.markdown('<p class="sec">💡 ADR Insights</p>', unsafe_allow_html=True)
    ia1,ia2,ia3 = st.columns(3)
    with ia1:
        st.markdown(f"""<div class="insight"><b>📅 Weekend ADR Premium</b><br>
        Weekend: <b>{fmt(daily[daily['stay_date'].dt.dayofweek>=5]['adr'].mean())}</b><br>
        Weekday: <b>{fmt(daily[daily['stay_date'].dt.dayofweek<5]['adr'].mean())}</b><br>
        Premium: <b>{(daily[daily['stay_date'].dt.dayofweek>=5]['adr'].mean()/
                       daily[daily['stay_date'].dt.dayofweek<5]['adr'].mean()-1)*100:+.1f}%</b>
        </div>""", unsafe_allow_html=True)
    with ia2:
        exp_conf = st.session_state.expanded[st.session_state.expanded["Cancellation_Status"]=="Confirmed"]
        adr_rt_avg = exp_conf.groupby("Room_Type")["Booked_Rate"].mean()
        st.markdown(f"""<div class="insight"><b>🛏️ ADR by Room Type</b><br>
        Standard: <b>{fmt(adr_rt_avg.get('Standard',0))}</b><br>
        Deluxe:   <b>{fmt(adr_rt_avg.get('Deluxe',0))}</b><br>
        Suite:    <b>{fmt(adr_rt_avg.get('Suite',0))}</b></div>""", unsafe_allow_html=True)
    with ia3:
        adr_ch = exp_conf.groupby("Booking_Channel")["Booked_Rate"].mean().sort_values(ascending=False)
        rows_str = "".join([f"{ch}: <b>{fmt(v)}</b><br>" for ch,v in adr_ch.items()])
        st.markdown(f"""<div class="insight"><b>📡 ADR by Channel</b><br>{rows_str}</div>""",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CANCELLATION RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab_cancel:
    st.markdown('<p class="sec">🚨 Cancellation Risk Predictor</p>', unsafe_allow_html=True)

    cm = cancel_metrics
    cc1,cc2,cc3,cc4 = st.columns(4)
    with cc1: st.markdown(kpi(f"{cm['AUC']:.4f}","AUC-ROC","o"), unsafe_allow_html=True)
    with cc2: st.markdown(kpi(f"{cm['Accuracy']*100:.1f}%","Accuracy","g"), unsafe_allow_html=True)
    with cc3:
        prec = cm["report"].get("1",{}).get("precision",0)
        st.markdown(kpi(f"{prec*100:.1f}%","Cancel Precision","r"), unsafe_allow_html=True)
    with cc4:
        recall = cm["report"].get("1",{}).get("recall",0)
        st.markdown(kpi(f"{recall*100:.1f}%","Cancel Recall","r"), unsafe_allow_html=True)

    st.markdown("""<div class="warn">
    ⚠️ <b>Note:</b> Cancellation patterns in this dataset appear near-random across features
    (AUC ≈ 0.51), which is common in synthetic or well-balanced hotel data.
    The model still provides calibrated probability scores useful for risk tiering.
    On real-world data with historical patterns, AUC typically reaches 0.75–0.85.
    </div>""", unsafe_allow_html=True)

    # ── Score a new booking ───────────────────────────────────────────────────
    st.markdown('<p class="sec">🎯 Score a New Booking</p>', unsafe_allow_html=True)
    st.caption("Enter booking details to get cancellation probability and predicted LOS.")

    col1, col2, col3 = st.columns(3)
    with col1:
        input_room    = st.selectbox("Room Type", ["Standard","Deluxe","Suite"])
        input_channel = st.selectbox("Booking Channel", CHANNELS)
        input_rate_plan = st.selectbox("Rate Plan", [
            "BAR","BAR + Member","Corporate","Corporate + Member",
            "Early Bird (> 30 days)","Early Bird (> 30 days) + Member",
            "Non-Refundable","Non-Refundable + Member"])
    with col2:
        input_checkin  = st.date_input("Check-in Date")
        input_checkout = st.date_input("Check-out Date",
            value=pd.Timestamp.today() + timedelta(days=3))
        input_booking  = st.date_input("Booking Date", value=pd.Timestamp.today())
    with col3:
        input_guests   = st.number_input("Number of Guests", 1, 6, 2)
        input_rate     = st.number_input("Booked Rate (IDR)", 1_000_000, 5_000_000, 1_200_000, 50_000)

    if st.button("🔍 Predict Cancellation Risk & LOS", type="primary"):
        from sklearn.preprocessing import LabelEncoder
        ci  = pd.Timestamp(input_checkin)
        co  = pd.Timestamp(input_checkout)
        bk  = pd.Timestamp(input_booking)
        nts = max((co - ci).days, 1)
        lt  = max((ci - bk).days, 0)

        # Encode categoricals using same logic as training
        room_enc    = {"Standard":0,"Deluxe":1,"Suite":2}
        chan_enc    = {"Direct":0,"OTA":1,"Walk-in":2,"Website":3}
        plan_list   = sorted(raw["Rate_Plan"].unique().tolist())
        plan_enc    = {p:i for i,p in enumerate(plan_list)}

        inp = pd.DataFrame([{
            "lead_time":           lt,
            "lead_time_sq":        lt**2,
            "Number_of_Nights":    nts,
            "Number_of_Guests":    input_guests,
            "nights_x_guests":     nts * input_guests,
            "Room_Type_enc":       room_enc.get(input_room, 0),
            "Booking_Channel_enc": chan_enc.get(input_channel, 0),
            "Rate_Plan_enc":       plan_enc.get(input_rate_plan, 0),
            "checkin_dow":         ci.dayofweek,
            "checkin_month":       ci.month,
            "checkin_quarter":     ci.quarter,
            "checkin_week":        ci.isocalendar()[1],
            "booking_dow":         bk.dayofweek,
            "booking_month":       bk.month,
            "booking_year":        bk.year,
            "is_member":           int("Member" in input_rate_plan),
            "is_early_bird":       int("Early Bird" in input_rate_plan),
            "is_non_refund":       int("Non-Refundable" in input_rate_plan),
            "is_bar":              int("BAR" in input_rate_plan),
            "is_corporate":        int("Corporate" in input_rate_plan),
            "is_weekend_ci":       int(ci.dayofweek >= 5),
            "Booked_Rate":         input_rate,
            "rev_per_guest":       input_rate / input_guests,
            "rev_per_night_guest": input_rate / (input_guests * nts),
            "month_sin":           np.sin(2*np.pi*ci.month/12),
            "month_cos":           np.cos(2*np.pi*ci.month/12),
        }])

        cancel_prob = float(m_cancel.predict_proba(inp[CANCEL_FEATS])[0, 1])
        los_pred    = int(np.clip(round(m_los.predict(inp[LOS_FEATS])[0]), 1, 7))

        risk_label = ("🟢 LOW"   if cancel_prob < 0.15 else
                      "🟡 MEDIUM" if cancel_prob < 0.30 else "🔴 HIGH")

        r1,r2,r3,r4 = st.columns(4)
        with r1: st.markdown(kpi(f"{cancel_prob*100:.1f}%","Cancel Probability",
                                   "g" if cancel_prob<0.15 else "o" if cancel_prob<0.3 else "r"),
                              unsafe_allow_html=True)
        with r2: st.markdown(kpi(risk_label,"Risk Level"), unsafe_allow_html=True)
        with r3: st.markdown(kpi(f"{los_pred} nights","Predicted Stay"), unsafe_allow_html=True)
        with r4: st.markdown(kpi(fmt(input_rate*los_pred),"Expected Revenue"), unsafe_allow_html=True)

        rec_msg = {
            "🟢 LOW":    "✅ Low cancellation risk. Standard confirmation appropriate.",
            "🟡 MEDIUM": "⚠️ Moderate risk. Consider requesting deposit or non-refundable upgrade.",
            "🔴 HIGH":   "🚨 High cancellation risk. Recommend non-refundable rate or overbooking buffer.",
        }
        st.markdown(f'<div class="insight">{rec_msg[risk_label]}</div>',
                    unsafe_allow_html=True)

    # ── Historical cancellation analysis ─────────────────────────────────────
    st.markdown('<p class="sec">📊 Historical Cancellation Analysis</p>', unsafe_allow_html=True)
    try:
        import altair as alt
        raw_plot = raw.copy()
        raw_plot["lead_bucket"] = pd.cut(raw_plot["lead_time"],
            bins=[0,7,30,60,90,365,730],
            labels=["0-7d","8-30d","31-60d","61-90d","91-365d","365d+"],
            right=True)

        ca1, ca2 = st.columns(2)
        with ca1:
            st.markdown("**Cancel Rate by Lead Time**")
            lt_data = (raw_plot.groupby("lead_bucket",observed=True)["is_cancelled"]
                       .mean().reset_index())
            lt_data["cancel_pct"] = lt_data["is_cancelled"] * 100
            ltc = (alt.Chart(lt_data).mark_bar(color="#e94560")
                   .encode(x=alt.X("lead_bucket:O",title="Lead Time"),
                           y=alt.Y("cancel_pct:Q",title="Cancel Rate %"),
                           tooltip=["lead_bucket:O",alt.Tooltip("cancel_pct:Q",format=".1f")])
                   .properties(height=250))
            st.altair_chart(ltc, use_container_width=True)

        with ca2:
            st.markdown("**Cancel Rate by Channel & Room Type**")
            cr_data = (raw_plot.groupby(["Booking_Channel","Room_Type"])["is_cancelled"]
                       .mean().reset_index())
            cr_data["cancel_pct"] = cr_data["is_cancelled"] * 100
            crc = (alt.Chart(cr_data).mark_bar()
                   .encode(x=alt.X("Booking_Channel:O",title="Channel"),
                           y=alt.Y("cancel_pct:Q",title="Cancel Rate %"),
                           color=alt.Color("Room_Type:N",scale=alt.Scale(
                               domain=["Standard","Deluxe","Suite"],
                               range=[ROOM_COLORS["Standard"],ROOM_COLORS["Deluxe"],ROOM_COLORS["Suite"]])),
                           tooltip=["Booking_Channel:O","Room_Type:N",
                                    alt.Tooltip("cancel_pct:Q",format=".1f")])
                   .properties(height=250))
            st.altair_chart(crc, use_container_width=True)

        # Monthly cancellation trend
        st.markdown("**Monthly Cancellation Trend**")
        raw_plot["ym"] = raw_plot["Check_in_Date"].dt.to_period("M").astype(str)
        mo_cancel = (raw_plot.groupby("ym").agg(
            cancel_rate=("is_cancelled","mean"),
            total=("is_cancelled","count")).reset_index())
        mo_cancel["cancel_pct"] = mo_cancel["cancel_rate"] * 100
        mc = (alt.Chart(mo_cancel).mark_line(point=True,color="#e94560",strokeWidth=2)
              .encode(x=alt.X("ym:O",title="Month"),
                      y=alt.Y("cancel_pct:Q",title="Cancel Rate %"),
                      tooltip=["ym:O",alt.Tooltip("cancel_pct:Q",format=".1f"),
                               alt.Tooltip("total:Q",title="Bookings")])
              .properties(height=220))
        st.altair_chart(mc, use_container_width=True)

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
    with l2: st.markdown(kpi(f"{lm['Accuracy']*100:.1f}%","Exact Acc.","g"), unsafe_allow_html=True)
    with l3: st.markdown(kpi(f"{raw['Number_of_Nights'].mean():.2f}","Hist Avg LOS","o"), unsafe_allow_html=True)
    with l4: st.markdown(kpi(f"{raw['Number_of_Nights'].median():.0f} nts","Median LOS"), unsafe_allow_html=True)

    try:
        import altair as alt

        la1, la2 = st.columns(2)

        with la1:
            st.markdown("**LOS Distribution (Actual vs Predicted)**")
            actual_counts = pd.Series(lm["test_actual"]).value_counts().sort_index().reset_index()
            actual_counts.columns = ["nights","count"]
            actual_counts["series"] = "Actual"
            pred_counts = pd.Series(lm["test_preds"]).value_counts().sort_index().reset_index()
            pred_counts.columns = ["nights","count"]
            pred_counts["series"] = "Predicted"
            los_dist = pd.concat([actual_counts, pred_counts])
            ldc = (alt.Chart(los_dist).mark_bar(opacity=0.7)
                   .encode(x=alt.X("nights:O",title="Nights"),
                           y=alt.Y("count:Q",title="Count"),
                           color=alt.Color("series:N",scale=alt.Scale(
                               domain=["Actual","Predicted"],range=["#0f3460","#e94560"])),
                           xOffset="series:N",
                           tooltip=["nights:O","series:N","count:Q"])
                   .properties(height=280))
            st.altair_chart(ldc, use_container_width=True)

        with la2:
            st.markdown("**Avg LOS by Rate Plan**")
            rp_los = (raw.groupby("Rate_Plan")["Number_of_Nights"]
                      .mean().reset_index().sort_values("Number_of_Nights", ascending=False))
            rp_los["Rate_Plan_short"] = rp_los["Rate_Plan"].str.replace(r" \(> 30 days\)","",regex=True)
            rpc = (alt.Chart(rp_los).mark_bar(color="#0f3460")
                   .encode(x=alt.X("Number_of_Nights:Q",title="Avg Nights"),
                           y=alt.Y("Rate_Plan_short:O",sort="-x",title="Rate Plan"),
                           tooltip=["Rate_Plan:O",
                                    alt.Tooltip("Number_of_Nights:Q",format=".2f")])
                   .properties(height=280))
            st.altair_chart(rpc, use_container_width=True)

        # LOS by check-in month
        st.markdown("**Avg LOS by Check-in Month & Channel**")
        raw["ci_month"] = raw["Check_in_Date"].dt.strftime("%b")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        los_mc = (raw.groupby(["ci_month","Booking_Channel"])["Number_of_Nights"]
                  .mean().reset_index())
        lmc = (alt.Chart(los_mc).mark_rect()
               .encode(x=alt.X("Booking_Channel:O",title="Channel"),
                       y=alt.Y("ci_month:O",sort=month_order,title="Month"),
                       color=alt.Color("Number_of_Nights:Q",
                           scale=alt.Scale(scheme="blues"),title="Avg Nights"),
                       tooltip=["ci_month:O","Booking_Channel:O",
                                alt.Tooltip("Number_of_Nights:Q",format=".2f")])
               .properties(height=300))
        st.altair_chart(lmc, use_container_width=True)

        # Revenue by LOS bucket
        st.markdown("**Revenue per Booking by LOS**")
        raw["los_bucket"] = raw["Number_of_Nights"].apply(
            lambda x: f"{x} night{'s' if x>1 else ''}")
        rev_los = (raw[raw["Cancellation_Status"]=="Confirmed"]
                   .groupby("Number_of_Nights")["Revenue_Generated"]
                   .mean().reset_index())
        rlc = (alt.Chart(rev_los).mark_bar(color="#27ae60")
               .encode(x=alt.X("Number_of_Nights:O",title="Nights"),
                       y=alt.Y("Revenue_Generated:Q",
                                axis=alt.Axis(format="~s"),title="Avg Revenue (IDR)"),
                       tooltip=["Number_of_Nights:O",
                                alt.Tooltip("Revenue_Generated:Q",format=",.0f")])
               .properties(height=260))
        st.altair_chart(rlc, use_container_width=True)

    except ImportError:
        pass

    # Optimization insight
    st.markdown('<p class="sec">🎯 LOS Optimization Recommendations</p>', unsafe_allow_html=True)
    conf = raw[raw["Cancellation_Status"]=="Confirmed"]
    rev_by_nights = conf.groupby("Number_of_Nights")["Revenue_Generated"].mean()
    best_los = rev_by_nights.idxmax()
    st.markdown(f"""<div class="insight">
    📌 <b>Highest avg revenue per booking:</b> <b>{best_los}-night stays</b>
       ({fmt(rev_by_nights[best_los])} avg revenue)<br>
    📌 <b>Most common LOS:</b> <b>{raw['Number_of_Nights'].mode()[0]} nights</b>
       ({(raw['Number_of_Nights']==raw['Number_of_Nights'].mode()[0]).mean()*100:.1f}% of bookings)<br>
    📌 <b>Recommendation:</b> Promote {best_los}-night minimum stay packages via BAR rates
       to maximize revenue per booking while maintaining occupancy.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CHANNEL MIX
# ══════════════════════════════════════════════════════════════════════════════
with tab_channel:
    st.markdown(f'<p class="sec">📡 Channel Mix — {horizon}-Day Forecast</p>',
                unsafe_allow_html=True)

    # KPIs: historical share
    conf_exp = st.session_state.expanded[st.session_state.expanded["Cancellation_Status"]=="Confirmed"]
    ch_share = conf_exp["Booking_Channel"].value_counts(normalize=True) * 100
    c1,c2,c3,c4 = st.columns(4)
    for col, ch in zip([c1,c2,c3,c4], CHANNELS):
        with col:
            st.markdown(kpi(f"{ch_share.get(ch,0):.1f}%", f"{ch} Share",
                             "g" if ch=="Direct" else ""), unsafe_allow_html=True)

    try:
        import altair as alt

        # Forecast channel rooms stacked area
        st.markdown("**Forecast: Rooms per Channel (Stacked)**")
        ch_fcast = fcast[["stay_date"]+CHANNELS].copy()
        ch_melt  = ch_fcast.melt("stay_date", var_name="Channel", value_name="Rooms")
        sac = (alt.Chart(ch_melt).mark_area(opacity=0.85)
               .encode(x=alt.X("stay_date:T",title="Date"),
                       y=alt.Y("Rooms:Q",stack="zero",title="Rooms"),
                       color=alt.Color("Channel:N",scale=alt.Scale(
                           domain=CHANNELS,
                           range=[CHANNEL_COLORS[c] for c in CHANNELS])),
                       tooltip=["stay_date:T","Channel:N",
                                alt.Tooltip("Rooms:Q",format=".1f")])
               .properties(title=f"Forecasted Rooms by Channel ({horizon} days)",height=280))
        st.altair_chart(sac, use_container_width=True)

        # Historical channel trend
        st.markdown("**Historical: Daily Rooms by Channel**")
        hist_ch = daily[["stay_date"]+CHANNELS].tail(180).melt(
            "stay_date", var_name="Channel", value_name="Rooms")
        hcc = (alt.Chart(hist_ch).mark_line(strokeWidth=1.5)
               .encode(x=alt.X("stay_date:T"),
                       y=alt.Y("Rooms:Q"),
                       color=alt.Color("Channel:N",scale=alt.Scale(
                           domain=CHANNELS,
                           range=[CHANNEL_COLORS[c] for c in CHANNELS])),
                       tooltip=["stay_date:T","Channel:N",
                                alt.Tooltip("Rooms:Q",format=".0f")])
               .properties(height=250))
        st.altair_chart(hcc, use_container_width=True)

        # ADR per channel over time
        st.markdown("**ADR per Channel (Historical)**")
        adr_ch_ts = (conf_exp.groupby(["stay_date","Booking_Channel"])["Booked_Rate"]
                     .mean().reset_index())
        acc = (alt.Chart(adr_ch_ts).mark_line(strokeWidth=1.5, opacity=0.8)
               .encode(x=alt.X("stay_date:T"),
                       y=alt.Y("Booked_Rate:Q",axis=alt.Axis(format="~s"),title="ADR (IDR)"),
                       color=alt.Color("Booking_Channel:N",scale=alt.Scale(
                           domain=CHANNELS,
                           range=[CHANNEL_COLORS[c] for c in CHANNELS])),
                       tooltip=["stay_date:T","Booking_Channel:N",
                                alt.Tooltip("Booked_Rate:Q",format=",.0f")])
               .properties(height=250))
        st.altair_chart(acc, use_container_width=True)

        # Channel revenue share pie-style bar
        st.markdown("**Revenue Contribution by Channel (Historical)**")
        ch_rev = (conf_exp.groupby("Booking_Channel")["Revenue_Generated"]
                  .sum().reset_index().sort_values("Revenue_Generated",ascending=False))
        ch_rev["pct"] = ch_rev["Revenue_Generated"] / ch_rev["Revenue_Generated"].sum() * 100
        crb = (alt.Chart(ch_rev).mark_bar()
               .encode(x=alt.X("Revenue_Generated:Q",axis=alt.Axis(format="~s"),title="Total Revenue"),
                       y=alt.Y("Booking_Channel:O",sort="-x",title="Channel"),
                       color=alt.Color("Booking_Channel:N",scale=alt.Scale(
                           domain=CHANNELS,
                           range=[CHANNEL_COLORS[c] for c in CHANNELS])),
                       tooltip=["Booking_Channel:O",
                                alt.Tooltip("Revenue_Generated:Q",format=",.0f"),
                                alt.Tooltip("pct:Q",format=".1f",title="Share %")])
               .properties(height=200))
        st.altair_chart(crb, use_container_width=True)

    except ImportError:
        st.line_chart(daily.set_index("stay_date")[CHANNELS])

    # Insights
    st.markdown('<p class="sec">💡 Channel Insights</p>', unsafe_allow_html=True)
    ci1,ci2,ci3 = st.columns(3)
    ch_adr  = conf_exp.groupby("Booking_Channel")["Booked_Rate"].mean()
    ch_canc = raw.groupby("Booking_Channel")["is_cancelled"].mean() * 100
    best_adr_ch = ch_adr.idxmax()
    low_canc_ch = ch_canc.idxmin()
    with ci1:
        rows_str = "".join([f"{ch}: <b>{fmt(v)}</b><br>" for ch,v in ch_adr.sort_values(ascending=False).items()])
        st.markdown(f"""<div class="insight"><b>💰 ADR by Channel</b><br>{rows_str}
        Best: <b>{best_adr_ch}</b></div>""", unsafe_allow_html=True)
    with ci2:
        rows_str = "".join([f"{ch}: <b>{v:.1f}%</b><br>" for ch,v in ch_canc.sort_values().items()])
        st.markdown(f"""<div class="insight"><b>🚨 Cancel Rate by Channel</b><br>{rows_str}
        Lowest risk: <b>{low_canc_ch}</b></div>""", unsafe_allow_html=True)
    with ci3:
        ch_los = raw[raw["Cancellation_Status"]=="Confirmed"].groupby("Booking_Channel")["Number_of_Nights"].mean()
        rows_str = "".join([f"{ch}: <b>{v:.2f} nts</b><br>" for ch,v in ch_los.sort_values(ascending=False).items()])
        st.markdown(f"""<div class="insight"><b>🛏️ Avg LOS by Channel</b><br>{rows_str}</div>""",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown('<p class="sec">📊 Model Diagnostics — Actual vs Predicted</p>',
                unsafe_allow_html=True)

    diag_target = st.selectbox("Select model to diagnose",
                                list(ts_metrics.keys()),
                                format_func=lambda x: LABEL_MAP.get(x, x))
    dm = ts_metrics[diag_target]
    diag_df = pd.DataFrame({
        "date":      pd.to_datetime(dm["test_dates"]),
        "Actual":    dm["test_actual"],
        "Predicted": dm["test_preds"],
    })
    diag_df["residual"] = diag_df["Actual"] - diag_df["Predicted"]
    diag_df["abs_pct"]  = (np.abs(diag_df["residual"]) /
                           np.where(diag_df["Actual"]==0, 1, diag_df["Actual"])) * 100

    d1,d2,d3,d4 = st.columns(4)
    with d1: st.markdown(kpi(f"{dm['MAE']:,.1f}","MAE"), unsafe_allow_html=True)
    with d2: st.markdown(kpi(f"{dm['MAPE']:.2f}%","MAPE","g"), unsafe_allow_html=True)
    with d3: st.markdown(kpi(f"{dm['R2']:.4f}","R²","g"), unsafe_allow_html=True)
    with d4: st.markdown(kpi(f"{diag_df['abs_pct'].median():.1f}%","Median Abs Err%"), unsafe_allow_html=True)

    try:
        import altair as alt
        melt = diag_df[["date","Actual","Predicted"]].melt("date")
        ch = (alt.Chart(melt).mark_line(strokeWidth=1.8)
              .encode(x=alt.X("date:T"),
                      y=alt.Y("value:Q",axis=alt.Axis(format="~s")),
                      color=alt.Color("variable:N",scale=alt.Scale(
                          domain=["Actual","Predicted"],range=["#0f3460","#e94560"])),
                      tooltip=["date:T",alt.Tooltip("value:Q",format=",.1f"),"variable:N"])
              .properties(height=300))
        st.altair_chart(ch, use_container_width=True)

        dc1,dc2 = st.columns(2)
        with dc1:
            st.markdown("**Residuals**")
            rc = (alt.Chart(diag_df).mark_bar(color="#e94560",opacity=.7)
                  .encode(x="date:T",
                          y=alt.Y("residual:Q",axis=alt.Axis(format="~s")),
                          tooltip=["date:T",alt.Tooltip("residual:Q",format=",.1f")])
                  .properties(height=220))
            st.altair_chart(rc, use_container_width=True)
        with dc2:
            st.markdown("**Abs % Error**")
            ec = (alt.Chart(diag_df.dropna()).mark_area(color="#0f3460",opacity=.5)
                  .encode(x="date:T",
                          y=alt.Y("abs_pct:Q",title="Abs % Error"),
                          tooltip=["date:T",alt.Tooltip("abs_pct:Q",format=".1f")])
                  .properties(height=220))
            st.altair_chart(ec, use_container_width=True)
    except ImportError:
        st.line_chart(diag_df.set_index("date")[["Actual","Predicted"]])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_fi:
    st.markdown('<p class="sec">🏷️ Feature Importance</p>', unsafe_allow_html=True)

    fi_source = st.radio("Model type", ["Time-Series","Booking-Level"], horizontal=True)

    if fi_source == "Time-Series":
        fi_target = st.selectbox("Select model", list(ts_models.keys()),
                                  format_func=lambda x: LABEL_MAP.get(x, x))
        fi_df = pd.DataFrame({
            "Feature":    feat_cols,
            "Importance": ts_models[fi_target].feature_importances_,
        }).sort_values("Importance",ascending=False).head(25).reset_index(drop=True)
    else:
        fi_bk_target = st.selectbox("Select model", ["Cancellation","LOS"])
        if fi_bk_target == "Cancellation":
            fi_df = pd.DataFrame({
                "Feature":    cancel_metrics["features"],
                "Importance": cancel_metrics["importances"],
            }).sort_values("Importance",ascending=False).head(20).reset_index(drop=True)
        else:
            fi_df = pd.DataFrame({
                "Feature":    los_metrics["features"],
                "Importance": los_metrics["importances"],
            }).sort_values("Importance",ascending=False).head(20).reset_index(drop=True)

    try:
        import altair as alt
        fic = (alt.Chart(fi_df).mark_bar(color="#0f3460")
               .encode(x="Importance:Q",
                       y=alt.Y("Feature:N",sort="-x"),
                       tooltip=["Feature:N","Importance:Q"])
               .properties(height=560))
        st.altair_chart(fic, use_container_width=True)
    except ImportError:
        st.bar_chart(fi_df.set_index("Feature")["Importance"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<p class="sec">📋 Daily Master Table</p>', unsafe_allow_html=True)

    view_cols = ["stay_date","nightly_revenue","rooms_occupied","occ_pct_total",
                 "occ_pct_Standard","occ_pct_Deluxe","occ_pct_Suite",
                 "adr","revpar"] + CHANNELS
    disp = daily[[c for c in view_cols if c in daily.columns]].copy()
    disp["stay_date"] = disp["stay_date"].dt.strftime("%Y-%m-%d")
    disp["nightly_revenue"] = disp["nightly_revenue"].apply(lambda x: f"{x:,.0f}")
    disp["adr"]   = disp["adr"].apply(lambda x: f"{x:,.0f}")
    disp["revpar"]= disp["revpar"].apply(lambda x: f"{x:,.0f}")
    for c in ["occ_pct_total","occ_pct_Standard","occ_pct_Deluxe","occ_pct_Suite"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda x: f"{x:.1f}%")
    st.dataframe(disp, use_container_width=True, height=400)

    st.markdown("**Expanded Stay-Night Table (sample 500)**")
    exp_show = st.session_state.expanded.head(500)[
        ["Booking_ID","stay_date","Check_in_Date","Check_out_Date",
         "Room_Type","Booking_Channel","Booked_Rate","Revenue_Generated",
         "Number_of_Nights","Cancellation_Status"]].copy()
    exp_show["stay_date"]     = exp_show["stay_date"].dt.strftime("%Y-%m-%d")
    exp_show["Check_in_Date"] = exp_show["Check_in_Date"].dt.strftime("%Y-%m-%d")
    exp_show["Check_out_Date"]= exp_show["Check_out_Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(exp_show, use_container_width=True, height=350)

    dl_daily = daily.copy(); dl_daily["stay_date"] = dl_daily["stay_date"].astype(str)
    st.download_button("⬇️ Download Daily Master CSV",
                        dl_daily.to_csv(index=False).encode(),
                        "daily_master.csv","text/csv")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#aaa;font-size:.75rem'>"
    "Hotel Revenue Intelligence Platform • 12 LightGBM Models • "
    "Revenue | Occupancy | ADR | Cancellation | LOS | Channel Mix • Streamlit"
    "</center>",
    unsafe_allow_html=True,
)
