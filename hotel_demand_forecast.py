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

from user_session import init_db, load_session, save_session, list_users, delete_session

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
    font-size:.88rem;line-height:1.6;color:#1a1a2e;}
  .warn{background:#fff8f0;border-left:3px solid #e67e22;
    border-radius:8px;padding:.85rem 1rem;margin:.4rem 0;font-size:.88rem;color:#1a1a2e;}
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
        for ch in CHANNELS:
            otb[ch] = 0.0
    else:
        exp = pd.DataFrame(rows)
        agg_dict = {
            "rooms_otb":   ("Booked_Rate", "count"),
            "revenue_otb": ("Booked_Rate", "sum"),
            "std_otb":     ("Room_Type", lambda x: (x == "Standard").sum()),
            "dlx_otb":     ("Room_Type", lambda x: (x == "Deluxe").sum()),
            "ste_otb":     ("Room_Type", lambda x: (x == "Suite").sum()),
            "adr_otb":     ("Booked_Rate", "mean"),
        }
        for ch in CHANNELS:
            agg_dict[ch] = ("Booking_Channel", lambda x, ch=ch: (x == ch).sum())
        otb = exp.groupby("stay_date").agg(**agg_dict).reset_index()
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
    df["doy"]       = df["stay_date"].dt.dayofyear
    # Cyclical encodings for seasonality
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)
    df["week_sin"]  = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"]  = np.cos(2 * np.pi * df["week"] / 52)
    df["doy_sin"]   = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["doy"] / 365)
    # Trend feature (days from start)
    df["days_from_start"] = (df["stay_date"] - df["stay_date"].min()).dt.days
    # Special periods
    df["is_month_start"] = (df["dom"] <= 5).astype(int)
    df["is_month_end"]   = (df["dom"] >= 25).astype(int)
    return df


def _add_lags(df, targets, max_lags=None):
    """
    Add lag and rolling features with optional cap on feature count.
    If max_lags is set, only include the most important lags to reduce overfitting.
    """
    df = df.copy()
    # Core lag set - most predictive lags (reduced from original to fight overfitting)
    core_lags = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 365]
    if max_lags:
        core_lags = core_lags[:max_lags]

    for t in targets:
        if t not in df.columns: continue
        # Core lag features (most predictive)
        for lag in core_lags:
            df[f"{t}_lag{lag}"] = df[t].shift(lag)
        # Essential rolling statistics (reduced set)
        for w in [7, 14, 30, 60]:
            df[f"{t}_roll{w}"] = df[t].shift(1).rolling(w).mean()
            df[f"{t}_roll{w}std"] = df[t].shift(1).rolling(w).std()
        # EWMA for recent trend emphasis
        for span in [7, 14, 30]:
            df[f"{t}_ewma{span}"] = df[t].shift(1).ewm(span=span, adjust=False).mean()
        # Key differentials
        for lag in [1, 7, 30]:
            df[f"{t}_diff{lag}"] = df[t].diff(lag)
            df[f"{t}_pct_chg{lag}"] = df[t].pct_change(lag).replace([np.inf, -np.inf], 0)
        # Momentum
        df[f"{t}_momentum"] = df[f"{t}_roll7"] - df[f"{t}_roll30"]
        # Seasonal comparison
        df[f"{t}_yoy"] = df[t].diff(365)
        df[f"{t}_mom"] = df[t].diff(30)
        df[f"{t}_wow"] = df[t].diff(7)
    return df


def _compute_r2(y_true, y_pred):
    """Compute R2 score safely handling edge cases."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def _walk_forward_cv(df, feat_cols, target, min_train_size=180, test_size=30, n_folds=5):
    """
    Perform expanding window walk-forward cross-validation.
    Returns dict with fold metrics and aggregated statistics.
    """
    from sklearn.metrics import mean_absolute_error

    n_samples = len(df)
    if n_samples < min_train_size + test_size * n_folds:
        # Not enough data - fall back to single split
        return None

    fold_metrics = []
    fold_importances = []

    for fold in range(n_folds):
        train_end = min_train_size + fold * test_size
        test_start = train_end
        test_end = min(train_end + test_size, n_samples)

        if test_end - test_start < 10:
            break

        X_train = df[feat_cols].iloc[:train_end]
        y_train = df[target].iloc[:train_end]
        X_test = df[feat_cols].iloc[test_start:test_end]
        y_test = df[target].iloc[test_start:test_end]

        # Skip if all same value
        if y_train.nunique() <= 1 or y_test.nunique() <= 1:
            continue

        # Light model for CV (simpler to avoid fold overfitting)
        import lightgbm as lgb
        m = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=16,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=42 + fold,
            verbose=-1,
        )

        m.fit(X_train, y_train)
        preds = np.clip(m.predict(X_test), 0, None)
        if "occ_pct" in target:
            preds = np.minimum(preds, 100.0)

        mae = mean_absolute_error(y_test, preds)
        denom = np.where(y_test.values == 0, 1, y_test.values)
        mape = float(np.mean(np.abs((y_test.values - preds) / denom)) * 100)
        r2 = float(_compute_r2(y_test.values, preds))

        fold_metrics.append({
            'fold': fold + 1,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'preds': preds,
            'actual': y_test.values,
            'dates': df["stay_date"].iloc[test_start:test_end].values,
        })
        fold_importances.append(m.feature_importances_)

    if not fold_metrics:
        return None

    # Aggregate fold metrics
    r2s = [f['r2'] for f in fold_metrics]
    maes = [f['mae'] for f in fold_metrics]
    mapes = [f['mape'] for f in fold_metrics]

    # Feature importance stability: variance across folds
    importances_arr = np.array(fold_importances)
    importance_mean = importances_arr.mean(axis=0)
    importance_std = importances_arr.std(axis=0)
    stability_score = np.where(importance_mean > 0,
                               1 - (importance_std / (importance_mean + 1e-8)),
                               0)

    return {
        'folds': fold_metrics,
        'r2_mean': float(np.mean(r2s)),
        'r2_std': float(np.std(r2s)),
        'r2_median': float(np.median(r2s)),
        'r2_min': float(np.min(r2s)),
        'r2_max': float(np.max(r2s)),
        'mae_mean': float(np.mean(maes)),
        'mape_mean': float(np.mean(mapes)),
        'importance_mean': importance_mean,
        'importance_stability': stability_score,
        'n_folds': len(fold_metrics),
    }


def _prune_features(df, feat_cols, target, cv_result, stability_threshold=0.5, importance_threshold=0.01):
    """
    Prune features based on importance stability across CV folds.
    Returns list of features to keep.
    """
    if cv_result is None or 'importance_stability' not in cv_result:
        return feat_cols

    stability = cv_result['importance_stability']
    importance = cv_result['importance_mean']

    # Normalize importance
    imp_sum = importance.sum()
    if imp_sum > 0:
        importance_norm = importance / imp_sum
    else:
        importance_norm = importance

    # Keep features that are either stable OR important
    keep_mask = (stability >= stability_threshold) | (importance_norm >= importance_threshold)

    # Always keep time-based features (they're stable and essential)
    time_features = ['dow', 'month', 'week', 'quarter', 'is_weekend', 'year', 'dom', 'doy',
                     'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'week_sin', 'week_cos',
                     'doy_sin', 'doy_cos', 'days_from_start', 'is_month_start', 'is_month_end']

    kept_features = []
    for i, feat in enumerate(feat_cols):
        if i < len(keep_mask):
            if keep_mask[i] or feat in time_features:
                kept_features.append(feat)
        elif feat in time_features:
            kept_features.append(feat)

    # Ensure we keep at least a minimum set of features
    if len(kept_features) < 10:
        # Fall back to top features by importance
        top_indices = np.argsort(importance)[-20:]
        for idx in top_indices:
            if feat_cols[idx] not in kept_features:
                kept_features.append(feat_cols[idx])

    return kept_features


# ══════════════════════════════════════════════════════════════════════════════
# TIME-SERIES MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_ts_models(daily: pd.DataFrame, use_walk_forward=True, prune_features=True):
    """
    Train time-series models with walk-forward validation and feature pruning.

    Args:
        daily: Daily aggregated DataFrame
        use_walk_forward: Whether to use walk-forward CV for robust evaluation
        prune_features: Whether to prune unstable features based on CV stability

    Returns:
        models: Dict of trained models per target
        metrics: Dict of metrics including robust CV stats
        feat_cols: List of feature columns used
        df: Processed DataFrame
        cv_results: Dict of walk-forward CV results per target
    """
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error

    all_ts_targets = TS_TARGETS + CHANNELS
    df = _add_time(daily)
    # Use reduced lag set to fight overfitting
    df = _add_lags(df, all_ts_targets, max_lags=8)
    df = df.dropna().reset_index(drop=True)

    non_feat = ({"stay_date"} | set(all_ts_targets) |
                {"rooms_occupied","rooms_Standard","rooms_Deluxe","rooms_Suite","revpar"} |
                {f"pct_{ch}" for ch in CHANNELS})
    feat_cols = [c for c in df.columns if c not in non_feat]

    # Final holdout split (always keep last 60 days for final evaluation)
    split = len(df) - 60
    models, metrics, cv_results = {}, {}, {}

    for target in all_ts_targets:
        if target not in df.columns: continue

        X = df[feat_cols]
        y = df[target]

        # Walk-forward cross-validation for robust evaluation
        cv_result = None
        if use_walk_forward and len(df) >= 300:
            cv_result = _walk_forward_cv(df, feat_cols, target, min_train_size=180, test_size=30, n_folds=5)
            cv_results[target] = cv_result

        # Feature pruning based on CV stability
        if prune_features and cv_result is not None:
            pruned_feats = _prune_features(df, feat_cols, target, cv_result,
                                           stability_threshold=0.3, importance_threshold=0.005)
            if len(pruned_feats) >= 10:
                feat_cols_to_use = pruned_feats
            else:
                feat_cols_to_use = feat_cols
        else:
            feat_cols_to_use = feat_cols

        Xtr, Xte = X.iloc[:split], X.iloc[split:]
        ytr, yte = y.iloc[:split], y.iloc[split:]

        # More conservative hyperparameters for better generalization
        if "occ_pct" in target:
            # Occupancy: stronger regularization for bounded target
            m = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=1500, learning_rate=0.02, max_depth=4, num_leaves=16,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.3, reg_lambda=0.5, random_state=42, verbose=-1,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            )
        elif target in CHANNELS:
            # Channel models: moderate complexity
            m = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=1200, learning_rate=0.03, max_depth=4, num_leaves=24,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.2, reg_lambda=0.3, random_state=42, verbose=-1,
            )
        else:
            # Revenue and ADR: balanced settings
            m = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=1500, learning_rate=0.03, max_depth=5, num_leaves=24,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.15, reg_lambda=0.25, random_state=42, verbose=-1,
            )

        # Train with early stopping on validation set
        m.fit(Xtr[feat_cols_to_use], ytr,
              eval_set=[(Xte[feat_cols_to_use], yte)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

        # Final evaluation on holdout
        preds = np.clip(m.predict(Xte[feat_cols_to_use]), 0, None)
        if "occ_pct" in target:
            preds = np.minimum(preds, 100.0)

        mae = mean_absolute_error(yte, preds)
        denom = np.where(yte.values == 0, 1, yte.values)
        mape = float(np.mean(np.abs((yte.values - preds) / denom)) * 100)
        r2 = float(_compute_r2(yte.values, preds))

        # Robust metrics from CV if available
        if cv_result:
            robust_r2 = cv_result['r2_median']
            robust_r2_std = cv_result['r2_std']
            expected_r2_range = (cv_result['r2_min'], cv_result['r2_max'])
        else:
            robust_r2 = r2
            robust_r2_std = 0
            expected_r2_range = (r2, r2)

        models[target] = m
        metrics[target] = {
            "MAE": mae, "MAPE": mape, "R2": r2,
            "Robust_R2": robust_r2,
            "Robust_R2_Std": robust_r2_std,
            "Expected_R2_Range": expected_r2_range,
            "test_preds": preds, "test_actual": yte.values,
            "test_dates": df["stay_date"].iloc[split:].values,
            "feat_cols": feat_cols_to_use,
            "n_features": len(feat_cols_to_use),
        }

    return models, metrics, feat_cols, df, cv_results


# ══════════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION & MODEL PROMOTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_drift_metrics(old_data: pd.DataFrame, new_data: pd.DataFrame,
                          numeric_cols=None, cat_cols=None) -> dict:
    """
    Compute drift metrics between old (training) and new (incoming) data.
    Returns dict with drift scores and warnings.
    """
    if numeric_cols is None:
        numeric_cols = ['Booked_Rate', 'Number_of_Nights', 'Number_of_Guests', 'lead_time']
    if cat_cols is None:
        cat_cols = ['Room_Type', 'Booking_Channel', 'Rate_Plan']

    drift_report = {
        'numeric': {},
        'categorical': {},
        'overall_drift_score': 0.0,
        'drift_level': 'low',  # low, medium, high
        'warnings': [],
    }

    # Numeric drift: standardized mean difference (Cohen's d)
    for col in numeric_cols:
        if col in old_data.columns and col in new_data.columns:
            old_vals = old_data[col].dropna()
            new_vals = new_data[col].dropna()
            if len(old_vals) > 0 and len(new_vals) > 0:
                old_mean, old_std = old_vals.mean(), old_vals.std()
                new_mean, new_std = new_vals.mean(), new_vals.std()
                pooled_std = np.sqrt((old_std**2 + new_std**2) / 2)
                if pooled_std > 0:
                    cohens_d = abs(new_mean - old_mean) / pooled_std
                else:
                    cohens_d = 0.0
                drift_report['numeric'][col] = {
                    'cohens_d': float(cohens_d),
                    'old_mean': float(old_mean),
                    'new_mean': float(new_mean),
                    'drift': 'high' if cohens_d > 1.0 else 'medium' if cohens_d > 0.5 else 'low',
                }

    # Categorical drift: PSI-like metric using Jensen-Shannon distance
    for col in cat_cols:
        if col in old_data.columns and col in new_data.columns:
            old_counts = old_data[col].value_counts(normalize=True)
            new_counts = new_data[col].value_counts(normalize=True)
            all_cats = set(old_counts.index) | set(new_counts.index)
            # Create aligned distributions
            old_dist = np.array([old_counts.get(c, 0) for c in all_cats])
            new_dist = np.array([new_counts.get(c, 0) for c in all_cats])
            # Add small epsilon to avoid log(0)
            old_dist = old_dist + 1e-10
            new_dist = new_dist + 1e-10
            # Jensen-Shannon distance
            m = 0.5 * (old_dist + new_dist)
            js_div = 0.5 * (np.sum(old_dist * np.log(old_dist / m)) +
                           np.sum(new_dist * np.log(new_dist / m)))
            js_dist = np.sqrt(js_div)
            drift_report['categorical'][col] = {
                'js_distance': float(js_dist),
                'drift': 'high' if js_dist > 0.3 else 'medium' if js_dist > 0.15 else 'low',
            }

    # Overall drift score (0-1 scale)
    num_scores = [d.get('cohens_d', 0) / 2 for d in drift_report['numeric'].values()]
    cat_scores = [d.get('js_distance', 0) for d in drift_report['categorical'].values()]
    all_scores = num_scores + cat_scores

    if all_scores:
        drift_report['overall_drift_score'] = float(np.mean(all_scores))
        max_score = max(all_scores)
        if max_score > 0.5 or drift_report['overall_drift_score'] > 0.3:
            drift_report['drift_level'] = 'high'
        elif max_score > 0.25 or drift_report['overall_drift_score'] > 0.15:
            drift_report['drift_level'] = 'medium'
        else:
            drift_report['drift_level'] = 'low'

    # Generate warnings
    high_drift_num = [c for c, d in drift_report['numeric'].items() if d.get('drift') == 'high']
    high_drift_cat = [c for c, d in drift_report['categorical'].items() if d.get('drift') == 'high']
    if high_drift_num:
        drift_report['warnings'].append(f"High numeric drift in: {', '.join(high_drift_num)}")
    if high_drift_cat:
        drift_report['warnings'].append(f"High categorical drift in: {', '.join(high_drift_cat)}")
    if drift_report['drift_level'] == 'high':
        drift_report['warnings'].append("Significant data drift detected. Model retraining strongly recommended.")

    return drift_report


def should_promote_new_model(old_metrics: dict, new_metrics: dict,
                             r2_tolerance=0.02) -> tuple:
    """
    Determine if new model should replace old model based on robust metrics.

    Returns: (should_promote: bool, reasons: list)
    """
    reasons = []

    # Compare robust R2 (median from walk-forward CV)
    old_r2     = old_metrics.get('Robust_R2', old_metrics.get('R2', 0))
    new_r2     = new_metrics.get('Robust_R2', new_metrics.get('R2', 0))
    new_r2_std = new_metrics.get('Robust_R2_Std', 0)

    old_mae = old_metrics.get('MAE', float('inf'))
    new_mae = new_metrics.get('MAE', float('inf'))

    promote = True

    # R² should not drop significantly
    if new_r2 < old_r2 - r2_tolerance:
        reasons.append(f"Robust R² dropped significantly ({new_r2:.3f} vs {old_r2:.3f})")
        promote = False

    # R² should be stable across folds — hard reject
    if new_r2_std > 0.50:
        reasons.append(f"High R² variance across validation folds ({new_r2_std:.3f})")
        promote = False

    # MAE should not regress catastrophically
    if old_mae > 0 and new_mae > old_mae * 2.0:
        reasons.append(f"MAE more than doubled ({new_mae:.2f} vs {old_mae:.2f})")
        promote = False

    # Minimum quality bar
    if new_r2 < 0.3:
        reasons.append(f"Robust R² below minimum threshold ({new_r2:.3f})")
        promote = False

    if promote:
        if new_r2 > old_r2 + 0.01:
            reasons.append(f"Improved robust R² ({new_r2:.3f} vs {old_r2:.3f})")
        elif old_mae > 0 and new_mae < old_mae * 0.9:
            reasons.append(f"Improved MAE ({new_mae:.2f} vs {old_mae:.2f})")
        else:
            reasons.append("Performance maintained within tolerance")

    return promote, reasons


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
# PICKUP MODEL v4 — replaces OTB-anchored forecast path
# ══════════════════════════════════════════════════════════════════════════════

PICKUP_SNAPSHOT_DAYS = [1, 2, 3, 5, 7, 10, 14, 21, 28, 35, 42, 50, 60, 75, 90]

PICKUP_FEATURES = [
    "days_to_arrival",
    "otb_occ_pct",
    "otb_rooms",
    "remaining_pct",
    "remaining_rooms",
    "fill_rate",          # NEW in v4
    "pace_7d",
    "pace_7d_pct",
    "pace_14d",
    "pace_14d_pct",
    "pace_30d",
    "pace_30d_pct",
    "otb_std",
    "otb_dlx",
    "otb_ste",
    "otb_adr",
    "dow",
    "month",
    "week",
    "quarter",
    "is_weekend",
    "dom",
    "doy",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
]

PICKUP_TARGETS = {
    "pickup_occ_pct":  "Pickup Occ %",
    "pickup_rooms":    "Pickup Rooms",
    "pickup_revenue":  "Pickup Revenue",
    "final_occ_pct":   "Final Occ % (total)",
    "final_revenue":   "Final Revenue (total)",
    "final_adr":       "Final ADR",
}

PROMOTION_GATE_TARGETS = ["pickup_occ_pct", "pickup_rooms", "pickup_revenue"]


def _compute_monthly_pace_baselines(
    conf: pd.DataFrame,
    capacity_total: int,
    lead_days_max: int = 90,
    min_month_stay_dates: int = 20,
) -> tuple[dict, float]:
    """
    Compute average daily booking pace per month for fill_rate calculations.
    
    Returns a tuple of (monthly_pace_dict, global_pace_fallback).
    monthly_pace maps month (1-12) to avg_daily_pace for that month.
    global_pace is used as fallback for months with sparse data.
    """
    conf = conf.copy()
    conf["Check_in_Date"] = pd.to_datetime(conf["Check_in_Date"])
    conf["Booking_Date"] = pd.to_datetime(conf["Booking_Date"])
    if "Check_out_Date" in conf.columns:
        conf["Check_out_Date"] = pd.to_datetime(conf["Check_out_Date"])
    else:
        conf["Check_out_Date"] = conf["Check_in_Date"] + timedelta(days=1)

    # Expand to stay-night granularity so pace aligns with stay_date targets.
    stay_rows = []
    for r in conf.itertuples(index=False):
        bdate = pd.Timestamp(r.Booking_Date)
        ci = pd.Timestamp(r.Check_in_Date)
        co = pd.Timestamp(r.Check_out_Date)
        for stay_date in pd.date_range(ci, co - timedelta(days=1)):
            lead = (pd.Timestamp(stay_date) - bdate).days
            if 0 <= lead <= lead_days_max:
                stay_rows.append({
                    "stay_date": pd.Timestamp(stay_date),
                    "month": pd.Timestamp(stay_date).month,
                })

    if not stay_rows:
        warnings.warn(
            "No valid stay-night leads found in pace baseline computation; using tiny global fallback pace."
        )
        return {}, 1e-6

    valid = pd.DataFrame(stay_rows)

    # Compute unique stay dates per month for normalization.
    monthly_stay_dates = valid.groupby("month")["stay_date"].nunique().to_dict()

    # Count valid stay-night leads per month.
    monthly_lead_counts = valid.groupby("month").size().to_dict()

    # Compute monthly pace: leads / (max_lead_days * unique_stay_dates).
    monthly_pace = {}
    for month, lead_count in monthly_lead_counts.items():
        n_stay_dates = max(monthly_stay_dates.get(month, 1), 1)
        monthly_pace[month] = lead_count / (lead_days_max * n_stay_dates)

    # Global fallback pace (average across valid stay-night lead-window data).
    global_stay_dates = max(valid["stay_date"].nunique(), 1)
    global_pace = len(valid) / (lead_days_max * global_stay_dates)

    # Guard sparse months by falling back to global pace.
    for month in list(monthly_pace.keys()):
        if monthly_stay_dates.get(month, 0) < min_month_stay_dates:
            monthly_pace[month] = global_pace

    # Sanity checks to catch silent denominator/filter regressions.
    invalid_months = {
        m: v for m, v in monthly_pace.items()
        if (not np.isfinite(v)) or v <= 1e-6 or v > 5.0
    }
    assert np.isfinite(global_pace) and 1e-6 < global_pace <= 5.0, (
        f"Suspicious global pace value: {global_pace}"
    )
    assert not invalid_months, f"Suspicious monthly pace values: {invalid_months}"
    
    return monthly_pace, global_pace


def _compute_inference_pace(
    raw: pd.DataFrame,
    as_of_date: pd.Timestamp,
    capacity_total: int,
) -> pd.DataFrame:
    """
    Compute real booking pace (7d, 14d, 30d) for all future stay dates.
    Mirrors the pace calculation in build_pickup_training_data() exactly,
    so there is no feature drift between train and inference.
    """
    conf = raw[raw["Cancellation_Status"] == "Confirmed"].copy()
    conf["Check_in_Date"]  = pd.to_datetime(conf["Check_in_Date"])
    conf["Check_out_Date"] = pd.to_datetime(conf["Check_out_Date"])
    conf["Booking_Date"]   = pd.to_datetime(conf["Booking_Date"])

    w30_start = as_of_date - timedelta(days=30)
    recent = conf[
        (conf["Booking_Date"] > w30_start) &
        (conf["Booking_Date"] <= as_of_date)
    ]

    rows = []
    for _, r in recent.iterrows():
        bdate = r["Booking_Date"]
        for d in pd.date_range(r["Check_in_Date"],
                               r["Check_out_Date"] - timedelta(days=1)):
            rows.append({"stay_date": pd.Timestamp(d), "Booking_Date": bdate})

    if not rows:
        return pd.DataFrame(columns=[
            "stay_date", "pace_7d", "pace_7d_pct",
            "pace_14d", "pace_14d_pct", "pace_30d", "pace_30d_pct",
        ])

    expanded  = pd.DataFrame(rows)
    w7_start  = as_of_date - timedelta(days=7)
    w14_start = as_of_date - timedelta(days=14)

    result = expanded.groupby("stay_date").apply(
        lambda g: pd.Series({
            "pace_7d":  (g["Booking_Date"] > w7_start).sum(),
            "pace_14d": (g["Booking_Date"] > w14_start).sum(),
            "pace_30d": len(g),
        })
    ).reset_index()

    result["pace_7d_pct"]  = result["pace_7d"]  / capacity_total * 100
    result["pace_14d_pct"] = result["pace_14d"] / capacity_total * 100
    result["pace_30d_pct"] = result["pace_30d"] / capacity_total * 100
    return result


def build_pickup_training_data(
    raw: pd.DataFrame,
    capacity: dict = None,
) -> pd.DataFrame:
    """
    Synthesise a pickup training table from raw booking-level data.

    For each (stay_date, snapshot_day) pair computes OTB state, booking pace,
    fill_rate, and calendar features. Target is pickup (can be negative = wash).
    """
    if capacity is None:
        capacity = CAPACITY

    conf = raw[raw["Cancellation_Status"] == "Confirmed"].copy()
    conf["Check_in_Date"]  = pd.to_datetime(conf["Check_in_Date"])
    conf["Check_out_Date"] = pd.to_datetime(conf["Check_out_Date"])
    conf["Booking_Date"]   = pd.to_datetime(conf["Booking_Date"])

    stay_rows = []
    for _, r in conf.iterrows():
        for d in pd.date_range(r["Check_in_Date"],
                               r["Check_out_Date"] - timedelta(days=1)):
            stay_rows.append({
                "stay_date":       d,
                "Booking_Date":    r["Booking_Date"],
                "Room_Type":       r["Room_Type"],
                "Booked_Rate":     r["Booked_Rate"],
                "Booking_Channel": r["Booking_Channel"],
            })
    stay_df = pd.DataFrame(stay_rows)

    finals = stay_df.groupby("stay_date").agg(
        final_rooms   = ("Booked_Rate", "count"),
        final_revenue = ("Booked_Rate", "sum"),
        final_std     = ("Room_Type",   lambda x: (x == "Standard").sum()),
        final_dlx     = ("Room_Type",   lambda x: (x == "Deluxe").sum()),
        final_ste     = ("Room_Type",   lambda x: (x == "Suite").sum()),
    ).reset_index()
    finals["final_occ_pct"] = finals["final_rooms"] / capacity["total"] * 100
    finals["final_adr"]     = finals["final_revenue"] / finals["final_rooms"].clip(lower=1)
    finals = finals[finals["final_rooms"] > 0].reset_index(drop=True)

    # Compute month-aware pace baselines for fill_rate (fix for seasonal flatness)
    monthly_pace, global_pace = _compute_monthly_pace_baselines(
        conf, capacity["total"], lead_days_max=90
    )

    records = []
    for stay_date in finals["stay_date"].values:
        stay_ts        = pd.Timestamp(stay_date)
        night_bookings = stay_df[stay_df["stay_date"] == stay_ts]

        for dta in PICKUP_SNAPSHOT_DAYS:
            as_of    = stay_ts - timedelta(days=dta)
            otb_mask = night_bookings["Booking_Date"] <= as_of
            otb_rows = night_bookings[otb_mask]

            otb_rooms   = len(otb_rows)
            otb_revenue = otb_rows["Booked_Rate"].sum()
            otb_std     = (otb_rows["Room_Type"] == "Standard").sum()
            otb_dlx     = (otb_rows["Room_Type"] == "Deluxe").sum()
            otb_ste     = (otb_rows["Room_Type"] == "Suite").sum()
            otb_occ_pct = otb_rooms / capacity["total"] * 100
            otb_adr     = otb_revenue / max(otb_rooms, 1)
            remaining   = max(0, capacity["total"] - otb_rooms)

            pace_7d  = len(night_bookings[
                (night_bookings["Booking_Date"] > as_of - timedelta(days=7)) &
                (night_bookings["Booking_Date"] <= as_of)
            ])
            pace_14d = len(night_bookings[
                (night_bookings["Booking_Date"] > as_of - timedelta(days=14)) &
                (night_bookings["Booking_Date"] <= as_of)
            ])
            pace_30d = len(night_bookings[
                (night_bookings["Booking_Date"] > as_of - timedelta(days=30)) &
                (night_bookings["Booking_Date"] <= as_of)
            ])

            # fill_rate: actual OTB vs expected at this DTA on the booking curve
            # Use month-aware pace for seasonal accuracy, with global fallback
            days_elapsed       = max(90 - dta, 0)
            month_pace         = monthly_pace.get(stay_ts.month, global_pace)
            expected_otb_rooms = month_pace * days_elapsed * capacity["total"]
            fill_rate          = otb_rooms / max(expected_otb_rooms, 1)
            fill_rate          = float(np.clip(fill_rate, 0.0, 8.0))

            # FIX 2: NO max(0,...) — negative pickup (wash) is valid training signal
            fin            = finals[finals["stay_date"] == stay_ts].iloc[0]
            pickup_rooms   = int(fin["final_rooms"])   - otb_rooms
            pickup_revenue = float(fin["final_revenue"]) - otb_revenue
            pickup_occ_pct = pickup_rooms / capacity["total"] * 100

            records.append({
                "stay_date":       stay_ts,
                "as_of_date":      as_of,
                "days_to_arrival": dta,
                "otb_rooms":       otb_rooms,
                "otb_occ_pct":     otb_occ_pct,
                "otb_revenue":     otb_revenue,
                "otb_adr":         otb_adr,
                "otb_std":         otb_std,
                "otb_dlx":         otb_dlx,
                "otb_ste":         otb_ste,
                "remaining_rooms": remaining,
                "remaining_pct":   remaining / capacity["total"] * 100,
                "fill_rate":       fill_rate,
                "pace_7d":         pace_7d,
                "pace_7d_pct":     pace_7d  / capacity["total"] * 100,
                "pace_14d":        pace_14d,
                "pace_14d_pct":    pace_14d / capacity["total"] * 100,
                "pace_30d":        pace_30d,
                "pace_30d_pct":    pace_30d / capacity["total"] * 100,
                "dow":             stay_ts.dayofweek,
                "month":           stay_ts.month,
                "week":            stay_ts.isocalendar()[1],
                "quarter":         stay_ts.quarter,
                "is_weekend":      int(stay_ts.dayofweek >= 5),
                "year":            stay_ts.year,
                "dom":             stay_ts.day,
                "doy":             stay_ts.dayofyear,
                "month_sin":       np.sin(2 * np.pi * stay_ts.month / 12),
                "month_cos":       np.cos(2 * np.pi * stay_ts.month / 12),
                "dow_sin":         np.sin(2 * np.pi * stay_ts.dayofweek / 7),
                "dow_cos":         np.cos(2 * np.pi * stay_ts.dayofweek / 7),
                "pickup_rooms":    pickup_rooms,
                "pickup_occ_pct":  pickup_occ_pct,
                "pickup_revenue":  pickup_revenue,
                "final_rooms":     int(fin["final_rooms"]),
                "final_occ_pct":   float(fin["final_occ_pct"]),
                "final_revenue":   float(fin["final_revenue"]),
                "final_adr":       float(fin["final_adr"]),
            })

    return pd.DataFrame(records).sort_values(
        ["stay_date", "days_to_arrival"]
    ).reset_index(drop=True)


def _walk_forward_cv_pickup(
    df: pd.DataFrame, feat_cols: list, target: str, n_folds: int = 5,
) -> dict | None:
    """Walk-forward CV on stay_date chronological order."""
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, r2_score

    unique_dates = df["stay_date"].sort_values().unique()
    fold_size    = max(1, len(unique_dates) // (n_folds + 1))
    fold_metrics = []
    fold_imps    = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_end  = min(fold_size * (fold + 2), len(unique_dates))
        if train_end >= len(unique_dates): break

        tr_dates = unique_dates[:train_end]
        te_dates = unique_dates[train_end:test_end]
        if len(te_dates) == 0: break

        tr = df[df["stay_date"].isin(tr_dates)]
        te = df[df["stay_date"].isin(te_dates)]
        if tr[target].nunique() <= 1 or te[target].nunique() <= 1: continue

        m = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=800, learning_rate=0.02,
            max_depth=6, num_leaves=31,
            min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.2,
            random_state=42 + fold, verbose=-1,
        )
        m.fit(tr[feat_cols], tr[target])
        preds = m.predict(te[feat_cols])

        mae  = mean_absolute_error(te[target], preds)
        r2   = float(r2_score(te[target].values, preds))
        bias = float(np.mean(preds - te[target].values))
        smape_denom = (np.abs(te[target].values) + np.abs(preds)) / 2 + 1e-8
        mape = float(np.mean(np.abs(te[target].values - preds) / smape_denom) * 100)

        fold_metrics.append({"fold": fold+1, "r2": r2, "mae": mae,
                              "mape": mape, "bias": bias,
                              "dates": te["stay_date"].values,
                              "preds": preds, "actual": te[target].values})
        fold_imps.append(m.feature_importances_)

    if not fold_metrics: return None

    imp_arr   = np.array(fold_imps)
    imp_mean  = imp_arr.mean(axis=0)
    imp_std   = imp_arr.std(axis=0)
    stability = np.where(imp_mean > 0, 1 - (imp_std / (imp_mean + 1e-8)), 0)

    r2s    = [f["r2"]   for f in fold_metrics]
    maes   = [f["mae"]  for f in fold_metrics]
    mapes  = [f["mape"] for f in fold_metrics]
    biases = [f["bias"] for f in fold_metrics]

    return {
        "folds": fold_metrics,
        "r2_mean": float(np.mean(r2s)),   "r2_std": float(np.std(r2s)),
        "r2_median": float(np.median(r2s)), "r2_min": float(np.min(r2s)),
        "r2_max": float(np.max(r2s)),
        "mae_mean": float(np.mean(maes)),  "mape_mean": float(np.mean(mapes)),
        "bias_mean": float(np.mean(biases)),
        "importance_mean": imp_mean, "importance_stability": stability,
        "n_folds": len(fold_metrics),
    }


def train_pickup_models(
    raw:              pd.DataFrame,
    use_walk_forward: bool = True,
    capacity:         dict = None,
) -> tuple:
    """Train pickup models. Returns (models, metrics, pickup_df, cv_results)."""
    if capacity is None:
        capacity = CAPACITY

    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, r2_score

    pickup_df    = build_pickup_training_data(raw, capacity=capacity)
    unique_dates = pickup_df["stay_date"].sort_values().unique()
    split_date   = unique_dates[int(len(unique_dates) * 0.8)]

    train_mask = pickup_df["stay_date"] <  split_date
    test_mask  = pickup_df["stay_date"] >= split_date
    X_train    = pickup_df.loc[train_mask, PICKUP_FEATURES]
    X_test     = pickup_df.loc[test_mask,  PICKUP_FEATURES]

    pickup_models  = {}
    pickup_metrics = {}
    cv_results     = {}

    for target, label in PICKUP_TARGETS.items():
        y_train = pickup_df.loc[train_mask, target]
        y_test  = pickup_df.loc[test_mask,  target]

        if y_train.nunique() <= 1:
            continue

        cv_result = None
        if use_walk_forward and len(X_train) >= 200:
            cv_result = _walk_forward_cv_pickup(
                pickup_df[train_mask].reset_index(drop=True),
                PICKUP_FEATURES, target,
            )
            cv_results[target] = cv_result

        # FIX 1: regression (MSE) not Tweedie — aligns loss with R metric
        # FIX 4: num_leaves = 2^(max_depth-1) to avoid train/test split fighting
        if "occ_pct" in target or "rooms" in target:
            params = dict(
                objective="regression",
                n_estimators=2000, learning_rate=0.015,
                max_depth=6, num_leaves=31,
                min_child_samples=25,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.2,
                random_state=42, verbose=-1,
            )
        else:
            params = dict(
                objective="regression",
                n_estimators=2000, learning_rate=0.015,
                max_depth=5, num_leaves=15,
                min_child_samples=25,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.2, reg_lambda=0.3,
                random_state=42, verbose=-1,
            )

        m = lgb.LGBMRegressor(**params)
        m.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

        # FIX 2: no clip at eval — measure true error including wash days
        preds = m.predict(X_test)
        if "occ_pct" in target:
            preds = np.minimum(preds, 100.0)

        mae  = mean_absolute_error(y_test, preds)
        r2   = float(r2_score(y_test.values, preds))
        bias = float(np.mean(preds - y_test.values))
        smape_denom = (np.abs(y_test.values) + np.abs(preds)) / 2 + 1e-8
        mape = float(np.mean(np.abs(y_test.values - preds) / smape_denom) * 100)

        robust_r2     = cv_result["r2_median"] if cv_result else r2
        robust_r2_std = cv_result["r2_std"]    if cv_result else 0.0
        r2_range      = (cv_result["r2_min"], cv_result["r2_max"]) if cv_result else (r2, r2)

        pickup_models[target]  = m
        pickup_metrics[target] = {
            "label":             label,
            "MAE":               mae,
            "MAPE":              mape,
            "R2":                r2,
            "Bias":              bias,
            "Robust_R2":         robust_r2,
            "Robust_R2_Std":     robust_r2_std,
            "Expected_R2_Range": r2_range,
            "test_preds":        preds,
            "test_actual":       y_test.values,
            "test_dates":        pickup_df.loc[test_mask, "stay_date"].values,
            "features":          PICKUP_FEATURES,
            "importances":       m.feature_importances_,
            "n_features":        len(PICKUP_FEATURES),
        }

    return pickup_models, pickup_metrics, pickup_df, cv_results


def forecast_with_pickup_models(
    pickup_models: dict,
    otb_df:        pd.DataFrame,
    as_of_date:    pd.Timestamp,
    raw:           pd.DataFrame,
    capacity:      dict = None,
    channels:      list = None,
    allow_wash:    bool = True,
) -> pd.DataFrame:
    """
    Forecast using pickup-trained models.
    Computes real pace before the loop (no zero-pace drift).
    Respects allow_wash for negative pickup handling.
    """
    if capacity is None: capacity = CAPACITY
    if channels is None: channels = CHANNELS

    # FIX 3 — real pace, computed once before the loop
    pace_df     = _compute_inference_pace(raw, as_of_date, capacity["total"])
    pace_lookup = pace_df.set_index("stay_date").to_dict("index") if len(pace_df) > 0 else {}

    # Compute month-aware pace baselines for fill_rate — mirrors training computation
    conf      = raw[raw["Cancellation_Status"] == "Confirmed"].copy()
    conf["Check_in_Date"] = pd.to_datetime(conf["Check_in_Date"])
    conf["Booking_Date"]  = pd.to_datetime(conf["Booking_Date"])
    monthly_pace, global_pace = _compute_monthly_pace_baselines(
        conf, capacity["total"], lead_days_max=90
    )

    records = []
    for _, row_otb in otb_df.reset_index(drop=True).iterrows():
        stay_date = pd.Timestamp(row_otb["stay_date"])
        dta       = (stay_date - as_of_date).days

        otb_rooms   = float(row_otb["rooms_otb"])
        otb_revenue = float(row_otb["revenue_otb"])
        otb_occ     = float(row_otb["occ_pct_otb"])
        otb_std     = float(row_otb["std_otb"])
        otb_dlx     = float(row_otb["dlx_otb"])
        otb_ste     = float(row_otb["ste_otb"])
        otb_adr     = float(row_otb["adr_otb"]) if row_otb["adr_otb"] > 0 else 0.0
        remaining   = float(row_otb["remaining_rooms"])

        pace_entry = pace_lookup.get(stay_date, {})
        pace_7d    = float(pace_entry.get("pace_7d",  0.0))
        pace_14d   = float(pace_entry.get("pace_14d", 0.0))
        pace_30d   = float(pace_entry.get("pace_30d", 0.0))

        days_elapsed       = max(90 - dta, 0)
        month_pace         = monthly_pace.get(stay_date.month, global_pace)
        expected_otb_rooms = month_pace * days_elapsed * capacity["total"]
        fill_rate          = otb_rooms / max(expected_otb_rooms, 1)
        fill_rate          = float(np.clip(fill_rate, 0.0, 8.0))

        feat_row = {
            "days_to_arrival": dta,
            "otb_occ_pct":     otb_occ,
            "otb_rooms":       otb_rooms,
            "remaining_pct":   remaining / capacity["total"] * 100,
            "remaining_rooms": remaining,
            "fill_rate":       fill_rate,
            "pace_7d":         pace_7d,
            "pace_7d_pct":     pace_7d  / capacity["total"] * 100,
            "pace_14d":        pace_14d,
            "pace_14d_pct":    pace_14d / capacity["total"] * 100,
            "pace_30d":        pace_30d,
            "pace_30d_pct":    pace_30d / capacity["total"] * 100,
            "otb_std":         otb_std,
            "otb_dlx":         otb_dlx,
            "otb_ste":         otb_ste,
            "otb_adr":         otb_adr,
            "dow":             stay_date.dayofweek,
            "month":           stay_date.month,
            "week":            stay_date.isocalendar()[1],
            "quarter":         stay_date.quarter,
            "is_weekend":      int(stay_date.dayofweek >= 5),
            "dom":             stay_date.day,
            "doy":             stay_date.dayofyear,
            "month_sin":       np.sin(2 * np.pi * stay_date.month / 12),
            "month_cos":       np.cos(2 * np.pi * stay_date.month / 12),
            "dow_sin":         np.sin(2 * np.pi * stay_date.dayofweek / 7),
            "dow_cos":         np.cos(2 * np.pi * stay_date.dayofweek / 7),
        }
        X = pd.DataFrame([feat_row])[PICKUP_FEATURES]

        pickup_lower = None if allow_wash else 0.0

        if "pickup_occ_pct" in pickup_models:
            pickup_occ_pct = float(np.clip(
                pickup_models["pickup_occ_pct"].predict(X)[0],
                pickup_lower, 100 - otb_occ,
            ))
        else:
            pickup_occ_pct = 0.0

        pickup_rooms = pickup_occ_pct * capacity["total"] / 100
        pickup_rooms = (max(pickup_rooms, -otb_rooms) if allow_wash
                        else min(max(pickup_rooms, 0.0), remaining))
        pickup_occ_pct = pickup_rooms / capacity["total"] * 100

        if "pickup_revenue" in pickup_models:
            pickup_revenue = float(np.clip(
                pickup_models["pickup_revenue"].predict(X)[0], pickup_lower, None
            ))
        else:
            pickup_revenue = pickup_rooms * (otb_adr if otb_adr > 0 else 0.0)

        total_rooms   = otb_rooms   + pickup_rooms
        total_occ     = total_rooms / capacity["total"] * 100
        total_revenue = otb_revenue + pickup_revenue
        total_adr     = total_revenue / max(total_rooms, 1)

        if "final_adr" in pickup_models:
            total_adr = float(np.clip(
                pickup_models["final_adr"].predict(X)[0], 0, None))
        if "final_revenue" in pickup_models:
            total_revenue = float(np.clip(
                pickup_models["final_revenue"].predict(X)[0], 0, None))
            total_adr = total_revenue / max(total_rooms, 1)

        rem_std   = float(row_otb["remaining_std"])
        rem_dlx   = float(row_otb["remaining_dlx"])
        rem_ste   = float(row_otb["remaining_ste"])
        rem_total = rem_std + rem_dlx + rem_ste
        if rem_total > 0 and pickup_rooms != 0:
            pickup_std = pickup_rooms * rem_std / rem_total
            pickup_dlx = pickup_rooms * rem_dlx / rem_total
            pickup_ste = pickup_rooms * rem_ste / rem_total
        else:
            pickup_std = pickup_dlx = pickup_ste = 0.0

        total_std_occ = min(max((row_otb["std_otb"] + pickup_std) / capacity["Standard"] * 100, 0), 100)
        total_dlx_occ = min(max((row_otb["dlx_otb"] + pickup_dlx) / capacity["Deluxe"]   * 100, 0), 100)
        total_ste_occ = min(max((row_otb["ste_otb"] + pickup_ste) / capacity["Suite"]    * 100, 0), 100)

        channel_preds = {}
        for ch in channels:
            channel_preds[ch] = (
                otb_rooms * (
                    otb_df[otb_df["stay_date"] == stay_date]
                    .get(ch, pd.Series([0])).values[0]
                ) / max(otb_rooms, 1)
            ) if otb_rooms > 0 else 0.0

        records.append({
            "stay_date":       stay_date,
            "otb_rooms":       otb_rooms,
            "otb_occ_pct":     otb_occ,
            "otb_revenue":     otb_revenue,
            "otb_adr":         otb_adr,
            "pickup_rooms":    pickup_rooms,
            "pickup_occ_pct":  pickup_occ_pct,
            "pickup_revenue":  pickup_revenue,
            "model_occ_pct":   total_occ,
            "model_revenue":   total_revenue,
            "total_rooms":     total_rooms,
            "total_occ_pct":   total_occ,
            "total_revenue":   total_revenue,
            "total_adr":       total_adr,
            "total_revpar":    total_revenue / capacity["total"],
            "remaining_rooms": remaining - pickup_rooms,
            "total_std_occ":   total_std_occ,
            "total_dlx_occ":   total_dlx_occ,
            "total_ste_occ":   total_ste_occ,
            **{ch: channel_preds.get(ch, 0) for ch in channels},
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# OTB-ANCHORED FORECAST ENGINE (legacy fallback)
# ══════════════════════════════════════════════════════════════════════════════

def forecast_otb_anchored(ts_models, feat_cols, ts_metrics, daily, otb_df, horizon=30):
    """
    OTB-Anchored Forecast:
    1. For each future date, check OTB (already confirmed)
    2. Model predicts TOTAL expected occupancy/revenue
    3. PICKUP = max(0, model_prediction - OTB)   [model may not exceed OTB if OTB>pred]
    4. TOTAL FORECAST = OTB + PICKUP (capped at capacity)

    Returns df with columns: stay_date, otb_*, model_*, pickup_*, total_*
    """
    # Build per-model feature column map (handles feature pruning per target)
    per_model_feat_cols = {}
    for t in ts_models.keys():
        if t in ts_metrics and "feat_cols" in ts_metrics[t]:
            per_model_feat_cols[t] = ts_metrics[t]["feat_cols"]
        else:
            per_model_feat_cols[t] = feat_cols
    all_ts_targets = TS_TARGETS + CHANNELS
    df       = _add_time(daily).sort_values("stay_date").reset_index(drop=True)

    # Get recent averages for initialization (last 14 days of historical data)
    recent   = df.tail(14)
    recent_means = {}
    for t in all_ts_targets + ["revpar","rooms_occupied",
                               "rooms_Standard","rooms_Deluxe","rooms_Suite"]:
        if t in recent.columns:
            recent_means[t] = recent[t].mean()
        else:
            recent_means[t] = 0.0
    for ch in CHANNELS:
        recent_means[f"pct_{ch}"] = recent[f"pct_{ch}"].mean() if f"pct_{ch}" in recent.columns else 25.0

    static_cols = [c for c in df.columns
                   if c not in set(all_ts_targets) |
                   {"stay_date","rooms_occupied","rooms_Standard","rooms_Deluxe",
                    "rooms_Suite","revpar"} |
                   {f"pct_{ch}" for ch in CHANNELS}
                   and "_lag" not in c and "_roll" not in c]

    # Build seasonal seed lookup: (month, dow) -> avg per target
    # Used to initialise future rows with seasonally-aware values rather than
    # a flat recent average, which prevents the recursive lag chain from
    # converging to the same constant for all far-future dates.
    _seed_targets = (all_ts_targets +
                     ["revpar", "rooms_occupied",
                      "rooms_Standard", "rooms_Deluxe", "rooms_Suite"] +
                     [f"pct_{ch}" for ch in CHANNELS])
    _df_seed = df.copy()
    _df_seed["_month"] = _df_seed["stay_date"].dt.month
    _df_seed["_dow"]   = _df_seed["stay_date"].dt.dayofweek
    _seasonal_seeds = {}
    for _t in _seed_targets:
        if _t in _df_seed.columns:
            _seasonal_seeds[_t] = _df_seed.groupby(["_month", "_dow"])[_t].mean()

    def _seed(t, date):
        """Return seasonal avg for (month, dow), falling back to recent mean."""
        key = (date.month, date.dayofweek)
        if t in _seasonal_seeds and key in _seasonal_seeds[t].index:
            val = _seasonal_seeds[t][key]
            return float(val) if not pd.isna(val) else recent_means.get(t, 0.0)
        return recent_means.get(t, 0.0)

    # Pre-extend history with future dates, initialized with recent averages
    # This ensures lag features have meaningful values, not zeros
    future_rows = []
    last_historical_date = df["stay_date"].max()
    forecast_start = otb_df["stay_date"].min()
    _min_date = df["stay_date"].min()  # for days_from_start computation

    # Fill gap between last historical date and forecast start if any
    gap_dates = pd.date_range(start=last_historical_date + timedelta(days=1),
                               end=forecast_start - timedelta(days=1), freq='D')
    for d in gap_dates:
        row = {"stay_date": d}
        for col in static_cols:
            if col in recent.columns:
                row[col] = recent[col].mean()
        # Override time features with correct values for this date
        _ts = _add_time(pd.DataFrame({"stay_date": [d]}))
        for _col in _ts.columns:
            if _col != "stay_date" and _col in row:
                row[_col] = _ts[_col].iloc[0]
        row["days_from_start"] = (d - _min_date).days
        for t in all_ts_targets + ["revpar","rooms_occupied",
                                   "rooms_Standard","rooms_Deluxe","rooms_Suite"]:
            row[t] = _seed(t, d)
        for ch in CHANNELS:
            row[f"pct_{ch}"] = _seed(f"pct_{ch}", d)
        future_rows.append(row)

    if future_rows:
        df = pd.concat([df, pd.DataFrame(future_rows)], ignore_index=True)
        df = df.sort_values("stay_date").reset_index(drop=True)

    history  = df.copy()

    records = []
    for i, row_otb in otb_df.reset_index(drop=True).iterrows():
        next_date = pd.Timestamp(row_otb["stay_date"])

        # Check if this date already exists (from gap filling)
        if next_date in history["stay_date"].values:
            # Update the existing row instead of creating new
            idx = history[history["stay_date"] == next_date].index[0]
        else:
            # Create new row with recent averages as seeds
            new_row = {"stay_date": next_date}
            for col in static_cols:
                if col in recent.columns:
                    new_row[col] = recent[col].mean()
            # Override time features with correct values for this future date
            _ts = _add_time(pd.DataFrame({"stay_date": [next_date]}))
            for _col in _ts.columns:
                if _col != "stay_date" and _col in new_row:
                    new_row[_col] = _ts[_col].iloc[0]
            new_row["days_from_start"] = (next_date - _min_date).days
            for t in all_ts_targets + ["revpar","rooms_occupied",
                                       "rooms_Standard","rooms_Deluxe","rooms_Suite"]:
                new_row[t] = _seed(t, next_date)
            for ch in CHANNELS:
                new_row[f"pct_{ch}"] = _seed(f"pct_{ch}", next_date)

            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
            idx = history.index[-1]

        # Recalculate all lag features with the full history
        history = _add_lags(history, all_ts_targets)
        # Re-find the index after _add_lags returns a new DataFrame
        idx = history[history["stay_date"] == next_date].index[0]

        # Build full feature row once - all models need consistent source data
        row_full = history.loc[[idx], feat_cols]

        # Fill any remaining NaNs with recent means instead of zeros
        for col in row_full.columns:
            if row_full[col].isna().any():
                if col in recent_means:
                    row_full[col] = row_full[col].fillna(recent_means[col])
                else:
                    row_full[col] = row_full[col].fillna(0)

        model_pred = {}
        for t, m in ts_models.items():
            # Use per-model feature columns (handles feature pruning)
            model_feats = per_model_feat_cols.get(t, feat_cols)
            row_f = row_full[model_feats]
            v = float(np.clip(m.predict(row_f)[0], 0, None))
            if "occ_pct" in t: v = min(v, 100.0)
            model_pred[t] = v
            history.at[idx, t] = v

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

        history.at[idx, "revpar"] = total_revenue / CAPACITY["total"]
        history.at[idx, "rooms_occupied"] = total_rooms

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
if "current_user" not in st.session_state:
    st.session_state.current_user = None
for k in ("ts_models","ts_metrics","ts_feat_cols","ts_df","daily",
          "m_cancel","cancel_metrics","m_los","los_metrics","raw","expanded",
          "cv_results","drift_report","pending_models","pending_metrics",
          "training_baseline_data","model_promotion_status",
          # Pickup v4 model artifacts
          "pickup_models","pickup_metrics","pickup_df","pickup_cv",
          "pending_pickup_models","pending_pickup_metrics"):
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
    st.markdown("## 👤 User Session")
    if st.session_state.current_user is None:
        username_input = st.text_input("Username", placeholder="Enter your username", key="username_in")
        if st.button("Load / Create session", type="primary", use_container_width=True, key="load_session_btn"):
            if username_input and username_input.strip():
                uname = username_input.strip()
                st.session_state.current_user = uname
                loaded = load_session(uname)
                if loaded:
                    for k, v in loaded.items():
                        st.session_state[k] = v
                    st.success(f"Loaded session for **{uname}**")
                else:
                    st.info(f"New session for **{uname}**. Upload data and train to save.")
                st.rerun()
            else:
                st.warning("Enter a username first.")
    else:
        st.success(f"**{st.session_state.current_user}**")
        if st.button("Switch User", use_container_width=True, key="switch_user_btn"):
            st.session_state.current_user = None
            for k in ("ts_models","ts_metrics","ts_feat_cols","ts_df","daily",
                      "m_cancel","cancel_metrics","m_los","los_metrics","raw","expanded"):
                st.session_state[k] = None
            st.rerun()
    if st.button("🗑️ Clear Session Data", use_container_width=True, key="clear_session_btn"):
        current_user = st.session_state.current_user
        if current_user:
            deleted = delete_session(current_user)
            if deleted:
                st.info(f"Deleted saved data for user: {current_user}")
        st.session_state.current_user = None
        for k in ("ts_models","ts_metrics","ts_feat_cols","ts_df","daily",
                  "m_cancel","cancel_metrics","m_los","los_metrics","raw","expanded",
                  "cv_results","drift_report","pending_models","pending_metrics",
                  "training_baseline_data","model_promotion_status"):
            st.session_state[k] = None
        st.success("Session cleared. Upload data and train to start fresh.")
        st.rerun()
    st.markdown("---")
    st.markdown("## ⚙️ Controls")
    uploaded = st.file_uploader("📂 Upload Bookings CSV", type=["csv"])
    st.markdown("---")

    st.markdown("### 📅 Forecast Settings")

    # as_of_date: default = last booking date in uploaded/trained data (simulated "today")
    st.markdown("**As-of Date** *(simulated 'today')*")
    st.caption("OTB uses confirmed bookings made on or before this date.")
    import datetime
    from dateutil.relativedelta import relativedelta

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

    # Generate month options starting from current month, extending 24 months out
    today = datetime.date.today()
    month_options = []
    month_values = {}  # maps display label to (year, month)
    for i in range(24):
        month_date = today + relativedelta(months=i)
        month_key = (month_date.year, month_date.month)
        # Calculate days from tomorrow to end of this month
        tomorrow = as_of_ts + timedelta(days=1)
        month_start = pd.Timestamp(month_date.replace(day=1))
        month_end = pd.Timestamp((month_date + relativedelta(months=1)).replace(day=1)) - timedelta(days=1)

        # Only show months that have at least some days in the future from as_of_date
        if month_end >= tomorrow:
            label = month_date.strftime("%B %Y")  # e.g., "February 2026"
            if i == 0:
                label += " (current month)"
            elif i == 1:
                label += " (next month)"
            month_options.append(label)
            month_values[label] = (month_date.year, month_date.month, month_start, month_end)

    default_month = month_options[0] if month_options else None
    selected_month_label = st.selectbox("Forecast through month", month_options, index=0)

    # Calculate horizon: days from tomorrow to end of selected month
    tomorrow = as_of_ts + timedelta(days=1)
    _, _, month_start, month_end = month_values[selected_month_label]
    horizon = (month_end - tomorrow).days + 1  # +1 to include the last day of month
    horizon = max(7, min(horizon, 365))  # clamp between 7 and 365 days

    st.caption(f"Forecast horizon: {horizon} days (through {month_end.strftime('%d %b %Y')})")

    with st.expander("🏗️ Room Capacity"):
        CAPACITY["Standard"] = st.number_input("Standard rooms", 1, 500, 50)
        CAPACITY["Deluxe"]   = st.number_input("Deluxe rooms",   1, 500, 25)
        CAPACITY["Suite"]    = st.number_input("Suite rooms",    1, 500, 10)
        CAPACITY["total"]    = CAPACITY["Standard"] + CAPACITY["Deluxe"] + CAPACITY["Suite"]

    st.markdown("---")

    # Drift detection warning for new data
    if uploaded is not None and st.session_state.raw is not None:
        try:
            df_peek = pd.read_csv(uploaded)
            df_peek["Booking_Date"] = pd.to_datetime(df_peek["Booking_Date"])
            drift = compute_drift_metrics(st.session_state.raw, df_peek)
            st.session_state.drift_report = drift
            uploaded.seek(0)

            if drift['drift_level'] == 'high':
                st.warning("⚠️ **High data drift detected!** Model retraining strongly recommended.")
            elif drift['drift_level'] == 'medium':
                st.info("ℹ️ Moderate data drift. Consider model retraining.")
        except Exception:
            pass

    # Model promotion status display
    if st.session_state.model_promotion_status:
        status = st.session_state.model_promotion_status
        if status.get('promoted'):
            st.success(f"✅ {status.get('message', 'Models promoted')}")
        else:
            st.error(f"❌ {status.get('message', 'Models rejected')}")
            if status.get('reasons'):
                for r in status['reasons']:
                    st.caption(f"• {r}")

    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        if uploaded is None:
            st.error("Upload a CSV first.")
        else:
            prog = st.progress(0, "Loading data…")
            raw_proc, expanded = load_and_expand(uploaded)
            prog.progress(25, "Building daily aggregates…")
            daily = build_daily(expanded)
            prog.progress(45, "Running walk-forward validation (TS models)…")
            ts_models, ts_metrics, feat_cols, ts_df, cv_results = train_ts_models(daily)
            prog.progress(65, "Training pickup models (v4)…")
            pickup_models, pickup_metrics, pickup_df, pickup_cv = train_pickup_models(
                raw_proc, capacity=CAPACITY
            )
            prog.progress(85, "Training booking-level models…")
            m_c, cancel_met, m_l, los_met = train_booking_models(raw_proc)
            prog.progress(100, "Evaluating model quality…")

            # Debug summary to explain quality-gate outcomes.
            if pickup_metrics:
                debug_rows = []
                for target, label in PICKUP_TARGETS.items():
                    if target in pickup_metrics:
                        met = pickup_metrics[target]
                        debug_rows.append({
                            "target": label,
                            "robust_r2": met.get("Robust_R2", met.get("R2", np.nan)),
                            "robust_r2_std": met.get("Robust_R2_Std", np.nan),
                            "mape": met.get("MAPE", np.nan),
                            "mae": met.get("MAE", np.nan),
                        })
                debug_metrics_df = pd.DataFrame(debug_rows)
            else:
                debug_metrics_df = pd.DataFrame()

            if pickup_df is not None and len(pickup_df) > 0:
                fill_debug_df = pickup_df.copy()
                fill_debug_df["month"] = fill_debug_df["stay_date"].dt.month
                fill_rate_monthly = fill_debug_df.groupby("month")["fill_rate"].agg(
                    count="count", mean="mean", std="std", min="min", max="max"
                ).reset_index().sort_values("month")
            else:
                fill_rate_monthly = pd.DataFrame()

            with st.expander("Debug: Pickup quality gate inputs", expanded=False):
                st.caption("Per-target metrics used by the promotion gate")
                if len(debug_metrics_df) > 0:
                    st.dataframe(debug_metrics_df, use_container_width=True)
                else:
                    st.caption("No pickup metrics available.")

                st.caption("Training fill_rate distribution by month")
                if len(fill_rate_monthly) > 0:
                    st.dataframe(fill_rate_monthly, use_container_width=True)
                else:
                    st.caption("No pickup training data available for fill_rate diagnostics.")

            # Model promotion decision (based on pickup model quality as primary)
            old_metrics = st.session_state.pickup_metrics
            promotion_reasons = []
            should_promote = True

            if old_metrics is not None and pickup_models:
                # Compare pickup model quality
                all_promotable = True
                for t in PROMOTION_GATE_TARGETS:
                    if t in old_metrics and t in pickup_metrics:
                        promote, reasons = should_promote_new_model(old_metrics[t], pickup_metrics[t])
                        if not promote:
                            all_promotable = False
                            promotion_reasons.extend([f"{PICKUP_TARGETS.get(t, t)}: {r}" for r in reasons])
                # Keep non-gating targets visible for diagnostics.
                non_gating_targets = [t for t in PICKUP_TARGETS.keys() if t not in PROMOTION_GATE_TARGETS]
                for t in non_gating_targets:
                    if t in old_metrics and t in pickup_metrics:
                        promote, reasons = should_promote_new_model(old_metrics[t], pickup_metrics[t])
                        if not promote:
                            promotion_reasons.extend([
                                f"[non-gating] {PICKUP_TARGETS.get(t, t)}: {r}" for r in reasons
                            ])
                should_promote = all_promotable

            elif pickup_models and pickup_metrics:
                # First training: apply minimum absolute quality bars
                # No old model to compare against, but don't auto-promote garbage
                for t in PROMOTION_GATE_TARGETS:
                    if t in pickup_metrics:
                        met = pickup_metrics[t]
                        r2 = met.get('Robust_R2', met.get('R2', 0))
                        r2_std = met.get('Robust_R2_Std', 0)
                        if r2 < 0.3:
                            should_promote = False
                            promotion_reasons.append(
                                f"{PICKUP_TARGETS.get(t, t)}: R² too low for first training ({r2:.3f})"
                            )
                        if r2_std > 0.50:
                            should_promote = False
                            promotion_reasons.append(
                                f"{PICKUP_TARGETS.get(t, t)}: R² variance too high for first training ({r2_std:.3f})"
                            )

            # Update session state with promotion decision
            if should_promote:
                st.session_state.ts_models = ts_models
                st.session_state.ts_metrics = ts_metrics
                st.session_state.ts_feat_cols = feat_cols
                st.session_state.ts_df = ts_df
                st.session_state.cv_results = cv_results
                st.session_state.daily = daily
                st.session_state.m_cancel = m_c
                st.session_state.cancel_metrics = cancel_met
                st.session_state.m_los = m_l
                st.session_state.los_metrics = los_met
                st.session_state.raw = raw_proc
                st.session_state.expanded = expanded
                # NEW: pickup model artifacts (v4 primary forecast path)
                st.session_state.pickup_models = pickup_models
                st.session_state.pickup_metrics = pickup_metrics
                st.session_state.pickup_df = pickup_df
                st.session_state.pickup_cv = pickup_cv
                st.session_state.training_baseline_data = raw_proc.copy()
                st.session_state.model_promotion_status = {
                    'promoted': True,
                    'message': 'Models promoted to production (pickup v4 primary)',
                    'reasons': promotion_reasons if promotion_reasons else ['Performance improved or maintained']
                }
                if st.session_state.current_user:
                    save_session(st.session_state.current_user, dict(st.session_state))
                prog.empty()
                st.success("✅ Models trained and promoted! (Pickup v4 active)")
            else:
                # Store as pending, don't replace current models
                st.session_state.pending_models = ts_models
                st.session_state.pending_metrics = ts_metrics
                st.session_state.pending_pickup_models = pickup_models
                st.session_state.pending_pickup_metrics = pickup_metrics
                st.session_state.model_promotion_status = {
                    'promoted': False,
                    'message': 'New models rejected - quality check failed',
                    'reasons': promotion_reasons
                }
                st.session_state.cv_results = cv_results
                st.session_state.pickup_cv = pickup_cv
                prog.empty()
                st.error("❌ New models failed quality check. Previous models retained.")

    if st.session_state.ts_models:
        st.markdown("### 📊 Model Quality (Robust OOS)")
        # Pickup v4 models (primary forecast path) - with Bias tracking
        if st.session_state.pickup_metrics:
            st.caption("**Pickup Models v4** (primary forecast)")
            for t, label in PICKUP_TARGETS.items():
                if t in st.session_state.pickup_metrics:
                    m = st.session_state.pickup_metrics[t]
                    robust_r2 = m.get('Robust_R2', m['R2'])
                    r2_range = m.get('Expected_R2_Range', (robust_r2, robust_r2))
                    bias = m.get('Bias', 0.0)
                    st.caption(f"**{label}** "
                              f"R²={robust_r2:.3f} (range: {r2_range[0]:.2f}-{r2_range[1]:.2f}) "
                              f"MAE={m['MAE']:.2f} Bias={bias:+.2f}")
        # Legacy TS models
        st.caption("**Time-Series Models** (legacy)")
        for t in TS_TARGETS:
            if t in st.session_state.ts_metrics:
                m = st.session_state.ts_metrics[t]
                robust_r2 = m.get('Robust_R2', m['R2'])
                r2_range = m.get('Expected_R2_Range', (robust_r2, robust_r2))
                st.caption(f"**{LABEL_MAP.get(t,t)}** "
                          f"R²={robust_r2:.3f} (range: {r2_range[0]:.2f}-{r2_range[1]:.2f}) "
                          f"MAPE={m['MAPE']:.1f}%")
        cm = st.session_state.cancel_metrics
        lm = st.session_state.los_metrics
        if cm: st.caption(f"**Cancellation** AUC={cm['AUC']:.3f}")
        if lm: st.caption(f"**LOS** MAE={lm['MAE']:.2f} nts Acc={lm['Accuracy']:.3f}")
        st.caption("12 LightGBM • Walk-forward CV • Feature pruning • Drift-aware • Pickup v4 with Bias tracking")

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
# NEW: pickup v4 models (primary forecast path)
pickup_models  = st.session_state.pickup_models
pickup_metrics = st.session_state.pickup_metrics

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE OTB + FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Computing OTB and generating forecast…"):
    otb_df  = get_otb(raw, as_of_ts, horizon)
    # Use pickup v4 models as primary forecast path (allow_wash=True default)
    if pickup_models:
        fcast = forecast_with_pickup_models(
            pickup_models, otb_df, as_of_ts,
            raw=raw, capacity=CAPACITY, channels=CHANNELS,
            allow_wash=True,
        )
    else:
        # Fallback to legacy OTB-anchored forecast
        fcast = forecast_otb_anchored(ts_models, feat_cols, ts_metrics, daily, otb_df, horizon)

tomorrow_str = (as_of_ts + timedelta(days=1)).strftime("%d %b %Y")
end_str      = (as_of_ts + timedelta(days=horizon)).strftime("%d %b %Y")
selected_month_name = selected_month_label.replace(" (current month)", "").replace(" (next month)", "")

# ══════════════════════════════════════════════════════════════════════════════
# TOP KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<p class="sec">📈 {selected_month_name} Forecast: {tomorrow_str} → {end_str} '
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

    # ── Month Filter for Results View ───────────────────────────────────────────
    months_in_forecast = sorted(fcast["stay_date"].dt.to_period("M").unique())
    month_labels = ["All Months"] + [str(m) for m in months_in_forecast]
    # Default to first month (current month) instead of "All Months"
    default_month_index = 1 if len(month_labels) > 1 else 0
    selected_view_month = st.selectbox("📅 Filter results by month", month_labels, index=default_month_index, key="month_filter_results")

    if selected_view_month != "All Months":
        selected_period = pd.Period(selected_view_month)
        fcast_filtered = fcast[fcast["stay_date"].dt.to_period("M") == selected_period].copy()
    else:
        fcast_filtered = fcast.copy()

    try:
        import altair as alt

        # ── Revenue: OTB + Pickup stacked bar + pickup line overlay ────────────────
        st.markdown("**Nightly Revenue: OTB (confirmed) vs Pickup (model prediction)**")
        rev_bar = fcast_filtered[["stay_date","otb_revenue","pickup_revenue"]].copy()
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

        # Pickup line overlay (offset to sit on top of OTB bars)
        pickup_line_rev = (alt.Chart(fcast_filtered)
                          .mark_line(color="#e67e22", strokeWidth=3, point=alt.MarkConfig(filled=True, size=60, color="#e67e22"))
                          .encode(x="stay_date:T",
                                  y=alt.Y("pickup_revenue:Q", title="Revenue (IDR)"),
                                  tooltip=["stay_date:T",
                                           alt.Tooltip("pickup_revenue:Q",format=",.0f",title="Pickup Revenue")]))

        # Total forecast line
        total_line_rev = (alt.Chart(fcast_filtered).mark_line(color="#3949ab",strokeDash=[4,2],strokeWidth=2)
                         .encode(x="stay_date:T",
                                 y=alt.Y("total_revenue:Q",axis=alt.Axis(format="~s")),
                                 tooltip=["stay_date:T",
                                          alt.Tooltip("total_revenue:Q",format=",.0f",title="Total Forecast")]))
        st.altair_chart(bar_rev + total_line_rev + pickup_line_rev, use_container_width=True)
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

        # ── Occupancy: OTB + Pickup with line overlay ────────────────
        st.markdown("**Occupancy %: OTB (confirmed) vs Pickup (model prediction)**")

        # Add pickup visibility indicator
        avg_pickup = fcast_filtered["pickup_occ_pct"].mean()
        if avg_pickup < 1.0:
            st.info(f"ℹ️ Average pickup is only {avg_pickup:.1f}%. This may be hard to see on the chart. The table below shows exact values.")

        occ_melt = fcast_filtered[["stay_date","otb_occ_pct","pickup_occ_pct"]].melt(
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

        # Pickup line with points for visibility
        pickup_line_occ = (alt.Chart(fcast_filtered)
                          .mark_line(color="#e67e22", strokeWidth=3,
                                     point=alt.MarkConfig(filled=True, size=70, color="#e67e22", stroke="#fff", strokeWidth=1))
                          .encode(x="stay_date:T",
                                  y=alt.Y("pickup_occ_pct:Q", scale=alt.Scale(domain=[0,105]), title="Occupancy %"),
                                  tooltip=["stay_date:T",
                                           alt.Tooltip("pickup_occ_pct:Q",format=".1f",title="Pickup Occ %")]))

        # Total occupancy line
        total_line_occ = (alt.Chart(fcast_filtered)
                         .mark_line(color="#3949ab",strokeDash=[4,2],strokeWidth=2)
                         .encode(x="stay_date:T",
                                 y=alt.Y("total_occ_pct:Q", scale=alt.Scale(domain=[0,105])),
                                 tooltip=["stay_date:T",
                                          alt.Tooltip("total_occ_pct:Q",format=".1f",title="Total Occ %")]))

        # 100% capacity line
        cap_line2 = (alt.Chart(pd.DataFrame({"stay_date":fcast_filtered["stay_date"],"cap":[100]*len(fcast_filtered)}))
                     .mark_rule(color="#e94560",strokeDash=[4,2],strokeWidth=1)
                     .encode(x="stay_date:T",y="cap:Q"))
        st.altair_chart(bar_occ + pickup_line_occ + total_line_occ + cap_line2, use_container_width=True)
        # Table: one row per stay_date with OTB % & Pickup % as columns
        occ_tbl = fcast_filtered[["stay_date", "otb_occ_pct", "pickup_occ_pct"]].copy()
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
        fcast_min = fcast_filtered["stay_date"].min()
        fcast_max = fcast_filtered["stay_date"].max()
        range_start = fcast_min - timedelta(days=90)
        hist_mask = (daily["stay_date"] >= range_start) & (daily["stay_date"] <= fcast_max)
        hist_occ = daily.loc[hist_mask, ["stay_date", "occ_pct_total"]].copy()
        hist_occ["series"] = "Historical (Actual)"
        fore_occ = fcast_filtered[["stay_date","total_occ_pct"]].rename(
            columns={"total_occ_pct":"occ_pct_total"}).copy()
        fore_occ["series"] = "Total Forecast"
        otb_line = fcast_filtered[["stay_date","otb_occ_pct"]].rename(
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
        fore_df = fcast_filtered[["stay_date", "otb_occ_pct", "total_occ_pct"]].rename(
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
        rt_melt = fcast_filtered[["stay_date","total_std_occ","total_dlx_occ","total_ste_occ"]].melt(
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
        st.line_chart(fcast_filtered.set_index("stay_date")[["otb_occ_pct","pickup_occ_pct","total_occ_pct"]])

    # Room type table: always show (outside try so it appears even if charts hit an error)
    try:
        rt_tbl = fcast_filtered[["stay_date", "total_std_occ", "total_dlx_occ", "total_ste_occ"]].copy()
        rt_tbl["stay_date"] = rt_tbl["stay_date"].dt.strftime("%Y-%m-%d")
        rt_tbl = rt_tbl.rename(columns={
            "stay_date": "Stay Date",
            "total_std_occ": "Standard",
            "total_dlx_occ": "Deluxe",
            "total_ste_occ": "Suite",
        })
        for c in ["Standard", "Deluxe", "Suite"]:
            rt_tbl[c] = rt_tbl[c].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
        with st.expander("📊 View occupancy by room type table"):
            st.dataframe(rt_tbl, use_container_width=True, height=300)
    except Exception as e:
        st.warning(f"Could not display room type table: {e}")

    # ── Per-date table ────────────────────────────────────────────────────────
    st.markdown('<p class="sec">📋 Daily Forecast Detail</p>', unsafe_allow_html=True)
    tbl = fcast_filtered[["stay_date","otb_rooms","otb_occ_pct","pickup_rooms","pickup_occ_pct",
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
    dl = fcast_filtered.copy(); dl["stay_date"] = dl["stay_date"].astype(str)
    month_suffix = selected_view_month.replace(' ', '_') if selected_view_month != "All Months" else selected_month_name.replace(' ', '_')
    st.download_button("⬇️ Download OTB-Anchored Forecast CSV",
                        dl.to_csv(index=False).encode(),
                        f"otb_forecast_{month_suffix}.csv","text/csv")

    # Insights
    st.markdown('<p class="sec">💡 Key Insights</p>', unsafe_allow_html=True)
    i1,i2,i3 = st.columns(3)
    with i1:
        high_otb = fcast_filtered.nlargest(3,"otb_occ_pct")[["stay_date","otb_occ_pct"]]
        rows_str = "".join([f"• {r['stay_date'].strftime('%d %b')}: <b>{r['otb_occ_pct']:.1f}%</b><br>"
                            for _,r in high_otb.iterrows()])
        st.markdown(f"""<div class="insight"><b>🟢 Strongest OTB Dates</b><br>{rows_str}</div>""",
                    unsafe_allow_html=True)
    with i2:
        low_otb = fcast_filtered.nsmallest(3,"otb_occ_pct")[["stay_date","otb_occ_pct","remaining_rooms"]]
        rows_str = "".join([f"• {r['stay_date'].strftime('%d %b')}: OTB {r['otb_occ_pct']:.1f}% "
                            f"({r['remaining_rooms']:.0f} rooms free)<br>"
                            for _,r in low_otb.iterrows()])
        st.markdown(f"""<div class="insight"><b>⚠️ Low OTB — Opportunity to Fill</b><br>{rows_str}</div>""",
                    unsafe_allow_html=True)
    with i3:
        wk   = fcast_filtered["stay_date"].dt.dayofweek >= 5
        st.markdown(f"""<div class="insight"><b>📅 Weekend vs Weekday</b><br>
        OTB: <b>{fcast_filtered[wk]['otb_occ_pct'].mean():.1f}%</b> vs
              <b>{fcast_filtered[~wk]['otb_occ_pct'].mean():.1f}%</b><br>
        Total: <b>{fcast_filtered[wk]['total_occ_pct'].mean():.1f}%</b> vs
               <b>{fcast_filtered[~wk]['total_occ_pct'].mean():.1f}%</b><br>
        Revenue: <b>{fmt(fcast_filtered[wk]['total_revenue'].mean())}</b> vs
                 <b>{fmt(fcast_filtered[~wk]['total_revenue'].mean())}</b>
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

        # Add top margin after KPI cards
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

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

        # Table: one row per stay_date with OTB ADR & Forecast ADR as columns
        adr_tbl = fcast[["stay_date", "otb_adr", "total_adr"]].copy()
        adr_tbl["stay_date"] = adr_tbl["stay_date"].dt.strftime("%Y-%m-%d")
        adr_tbl = adr_tbl.rename(columns={
            "stay_date": "Stay Date",
            "otb_adr": "OTB ADR",
            "total_adr": "Forecast ADR",
        })
        adr_tbl["OTB ADR"] = adr_tbl["OTB ADR"].apply(lambda x: f"{x:,.0f}")
        adr_tbl["Forecast ADR"] = adr_tbl["Forecast ADR"].apply(lambda x: f"{x:,.0f}")
        with st.expander("📊 View ADR table (OTB ADR vs Forecast ADR by date)"):
            st.dataframe(adr_tbl, use_container_width=True, height=300)

        # Historical ADR + forecast
        # Show actuals for the full range that overlaps the chart: from (forecast start − 90d) to forecast end,
        # so April–Sep and all other actuals in the forecast window appear (not just the last 90 rows).
        fcast_min = fcast["stay_date"].min()
        fcast_max = fcast["stay_date"].max()
        range_start = fcast_min - timedelta(days=90)
        hist_mask = (daily["stay_date"] >= range_start) & (daily["stay_date"] <= fcast_max)
        h_adr = daily.loc[hist_mask, ["stay_date", "adr"]].copy()
        h_adr["series"] = "Historical"
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

        # Table: one row per date with Historical ADR and Forecast ADR
        all_dates_adr = pd.DataFrame({"stay_date": pd.date_range(start=range_start, end=fcast_max, freq="D")})
        hist_adr_df = daily.loc[hist_mask, ["stay_date", "adr"]].rename(columns={"adr": "Historical ADR"})
        fore_adr_df = fcast[["stay_date", "total_adr"]].rename(columns={"total_adr": "Forecast ADR"})
        adr_hf_tbl = all_dates_adr.merge(hist_adr_df, on="stay_date", how="left").merge(fore_adr_df, on="stay_date", how="left")
        adr_hf_tbl["stay_date"] = adr_hf_tbl["stay_date"].dt.strftime("%Y-%m-%d")
        adr_hf_tbl = adr_hf_tbl.rename(columns={"stay_date": "Stay Date"})
        for c in ["Historical ADR", "Forecast ADR"]:
            adr_hf_tbl[c] = adr_hf_tbl[c].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        with st.expander("📊 View historical + forecast ADR table"):
            st.dataframe(adr_hf_tbl, use_container_width=True, height=350)

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

        # Table: one row per stay_date with OTB ADR & Forecast ADR as columns
        adr_tbl = fcast[["stay_date", "otb_adr", "total_adr"]].copy()
        adr_tbl["stay_date"] = adr_tbl["stay_date"].dt.strftime("%Y-%m-%d")
        adr_tbl = adr_tbl.rename(columns={
            "stay_date": "Stay Date",
            "otb_adr": "OTB ADR",
            "total_adr": "Forecast ADR",
        })
        adr_tbl["OTB ADR"] = adr_tbl["OTB ADR"].apply(lambda x: f"{x:,.0f}")
        adr_tbl["Forecast ADR"] = adr_tbl["Forecast ADR"].apply(lambda x: f"{x:,.0f}")
        with st.expander("📊 View ADR table (OTB ADR vs Forecast ADR by date)"):
            st.dataframe(adr_tbl, use_container_width=True, height=300)

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
    st.markdown('<p class="sec">📊 Model Diagnostics (Robust OOS)</p>', unsafe_allow_html=True)
    diag_t = st.selectbox("Select model",list(ts_metrics.keys()),
                           format_func=lambda x:LABEL_MAP.get(x,x))
    dm = ts_metrics[diag_t]
    cv = st.session_state.cv_results.get(diag_t) if st.session_state.cv_results else None

    # Robust metrics display with fold information
    diag_df = pd.DataFrame({
        "date": pd.to_datetime(dm["test_dates"]),
        "Actual": dm["test_actual"],
        "Predicted": dm["test_preds"],
    })
    diag_df["residual"] = diag_df["Actual"] - diag_df["Predicted"]
    diag_df["abs_pct"]  = (np.abs(diag_df["residual"]) /
                            np.where(diag_df["Actual"]==0,1,diag_df["Actual"]))*100

    # Main metrics row
    d1,d2,d3,d4 = st.columns(4)
    with d1: st.markdown(kpi(f"{dm['MAE']:,.1f}","MAE"), unsafe_allow_html=True)
    with d2: st.markdown(kpi(f"{dm['MAPE']:.2f}%","MAPE","g"), unsafe_allow_html=True)
    with d3:
        robust_r2 = dm.get('Robust_R2', dm['R2'])
        st.markdown(kpi(f"{robust_r2:.4f}","Robust R²","g"), unsafe_allow_html=True)
    with d4: st.markdown(kpi(f"{diag_df['abs_pct'].median():.1f}%","Median Abs Err%"), unsafe_allow_html=True)

    # Expected R2 range from walk-forward CV
    r2_range = dm.get('Expected_R2_Range', (robust_r2, robust_r2))
    r2_std = dm.get('Robust_R2_Std', 0)

    st.markdown(f"""
    <div style="background:#f8f9ff;border-left:3px solid #0f3460;border-radius:8px;padding:.85rem 1rem;margin:.5rem 0;">
    <b>Expected R² Range:</b> {r2_range[0]:.3f} – {r2_range[1]:.3f} &nbsp;|&nbsp;
    <b>Cross-fold Std:</b> {r2_std:.3f} &nbsp;|&nbsp;
    <b>Features Used:</b> {dm.get('n_features', 'N/A')}
    </div>
    """, unsafe_allow_html=True)

    # Walk-forward CV fold details
    if cv and 'folds' in cv:
        st.markdown('<p class="sec">📈 Walk-Forward Validation Folds</p>', unsafe_allow_html=True)
        fold_data = []
        for fold in cv['folds']:
            fold_data.append({
                'Fold': fold['fold'],
                'Test Period': f"{pd.Timestamp(fold['dates'][0]).strftime('%Y-%m-%d')} to {pd.Timestamp(fold['dates'][-1]).strftime('%Y-%m-%d')}",
                'R²': f"{fold['r2']:.3f}",
                'MAE': f"{fold['mae']:.1f}",
                'MAPE': f"{fold['mape']:.1f}%",
            })
        fold_df = pd.DataFrame(fold_data)
        st.dataframe(fold_df, use_container_width=True, hide_index=True)

        # Fold R2 distribution chart
        try:
            import altair as alt
            fold_chart_df = pd.DataFrame({
                'Fold': [f"Fold {f['fold']}" for f in cv['folds']],
                'R2': [f['r2'] for f in cv['folds']],
                'MAPE': [f['mape'] for f in cv['folds']],
            })
            fc1, fc2 = st.columns(2)
            with fc1:
                st.altair_chart(alt.Chart(fold_chart_df).mark_bar(color="#0f3460")
                    .encode(x=alt.X("Fold:N"), y=alt.Y("R2:Q", scale=alt.Scale(domain=[0, 1])),
                            tooltip=["Fold:N", alt.Tooltip("R2:Q", format=".3f")])
                    .properties(height=220, title="R² by Fold"), use_container_width=True)
            with fc2:
                st.altair_chart(alt.Chart(fold_chart_df).mark_bar(color="#e94560")
                    .encode(x=alt.X("Fold:N"), y=alt.Y("MAPE:Q"),
                            tooltip=["Fold:N", alt.Tooltip("MAPE:Q", format=".1f")])
                    .properties(height=220, title="MAPE by Fold"), use_container_width=True)
        except ImportError:
            pass

    # Actual vs Predicted and Residuals
    st.markdown('<p class="sec">📉 Prediction Quality</p>', unsafe_allow_html=True)
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

    # Drift report if available
    if st.session_state.drift_report:
        st.markdown('<p class="sec">🔄 Data Drift Analysis</p>', unsafe_allow_html=True)
        drift = st.session_state.drift_report
        drift_level = drift.get('drift_level', 'low')
        drift_color = {'low': '#27ae60', 'medium': '#e67e22', 'high': '#e74c3c'}
        st.markdown(f"""
        <div style="background:#fff;border-left:4px solid {drift_color.get(drift_level, '#27ae60')};border-radius:8px;padding:.85rem 1rem;margin:.5rem 0;">
        <b>Overall Drift Level:</b> <span style="color:{drift_color.get(drift_level)};font-weight:bold;text-transform:uppercase;">{drift_level}</span> &nbsp;|&nbsp;
        <b>Drift Score:</b> {drift.get('overall_drift_score', 0):.3f}
        </div>
        """, unsafe_allow_html=True)
        if drift.get('warnings'):
            for warning in drift['warnings']:
                st.warning(warning)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_fi:
    st.markdown('<p class="sec">🏷️ Feature Importance</p>', unsafe_allow_html=True)
    fi_src = st.radio("Model type",["Time-Series","Booking-Level"],horizontal=True)
    if fi_src == "Time-Series":
        fi_t = st.selectbox("Model",list(ts_models.keys()),
                             format_func=lambda x:LABEL_MAP.get(x,x))
        # Use per-model feature columns (handles feature pruning)
        model_feat_cols = ts_metrics.get(fi_t, {}).get("feat_cols", feat_cols)
        fi_df = pd.DataFrame({"Feature":model_feat_cols,
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

    st.markdown(f"**On-the-Book as of {as_of_date} (through {selected_month_name})**")
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
                        f"otb_forecast_{as_of_date}_{selected_month_name.replace(' ', '_')}.csv","text/csv")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#aaa;font-size:.75rem'>"
    "Hotel Revenue Intelligence • OTB-Anchored • 12 LightGBM Models • "
    "Revenue | Occupancy | ADR | Cancellation | LOS | Channel Mix</center>",
    unsafe_allow_html=True,
)
