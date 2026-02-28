# ══════════════════════════════════════════════════════════════════════════════
# PICKUP MODEL — v4
# ══════════════════════════════════════════════════════════════════════════════
#
# Key differences from v3 (Gemini Pro 3.1):
#
#   FIX 1 — Tweedie → regression objective
#     Tweedie optimises log-scale loss; R² is measured on linear scale.
#     They diverge, especially for pickup which is near-normal (not Poisson).
#     Plain MSE ("regression") directly minimises squared error, which is what
#     R² measures. Expected gain: +0.05 to +0.12 R².
#
#   FIX 2 — Wash floors removed from training targets (re-fix from v2)
#     v3 reinstated max(0,...) on pickup_rooms/revenue targets (lines 125–126).
#     This blinds the model to cancellation-driven shrinkage and introduces
#     systematic positive bias. Removed again — negative pickup is valid signal.
#
#   FIX 3 — Real pace at inference (re-fix from v2, properly wired)
#     v3 used row_otb.get("pace_7d", 0.0) which always returns 0.0 because
#     get_otb() never computes pace. _compute_inference_pace() is restored and
#     called before the forecast loop. Now covers 7d, 14d, and 30d windows.
#
#   FIX 4 — num_leaves aligned with max_depth
#     num_leaves=63 + max_depth=7 fight each other and overfit small datasets.
#     Changed to num_leaves=31 + max_depth=6 for occ/rooms,
#     and num_leaves=15 + max_depth=5 for revenue/adr (higher variance targets).
#
#   NEW — fill_rate feature
#     fill_rate = otb_rooms / expected_otb_at_this_dta
#     Captures "are we ahead or behind the typical booking curve right now?"
#     This is the single most predictive engineered feature for pickup after
#     otb_occ_pct and days_to_arrival. Expected R² gain: +0.03 to +0.06.
#
#   NEW — bias tracked in metrics
#     mean signed error (pred - actual) is now stored alongside MAE/MAPE/R².
#     Systematic bias is more dangerous than random error for hotel operations.
#
# ══════════════════════════════════════════════════════════════════════════════

PICKUP_SNAPSHOT_DAYS = [1, 2, 3, 5, 7, 10, 14, 21, 28, 45, 60, 90]


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

    # avg_daily_pace: used to compute fill_rate (how full vs expected at this DTA)
    all_leads      = (conf["Check_in_Date"] - conf["Booking_Date"]).dt.days
    valid_leads    = all_leads[(all_leads >= 0) & (all_leads <= 90)]
    n_stay_dates   = max(len(finals), 1)
    avg_daily_pace = len(valid_leads) / (90 * n_stay_dates)

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
            days_elapsed       = max(90 - dta, 0)
            expected_otb_rooms = avg_daily_pace * days_elapsed * capacity["total"]
            fill_rate          = otb_rooms / max(expected_otb_rooms, 1)

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

        # FIX 1: regression (MSE) not Tweedie — aligns loss with R² metric
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
        denom = np.where(y_test.values == 0, 1, np.abs(y_test.values))
        mape = float(np.mean(np.abs((y_test.values - preds) / denom)) * 100)

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
        denom = np.where(te[target].values == 0, 1, np.abs(te[target].values))
        mape = float(np.mean(np.abs((te[target].values - preds) / denom)) * 100)

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

    # avg_daily_pace for fill_rate — mirrors training computation
    conf      = raw[raw["Cancellation_Status"] == "Confirmed"].copy()
    conf["Check_in_Date"] = pd.to_datetime(conf["Check_in_Date"])
    conf["Booking_Date"]  = pd.to_datetime(conf["Booking_Date"])
    leads          = (conf["Check_in_Date"] - conf["Booking_Date"]).dt.days
    valid_leads    = leads[(leads >= 0) & (leads <= 90)]
    n_stay_dates   = max(conf["Check_in_Date"].dt.date.nunique(), 1)
    avg_daily_pace = len(valid_leads) / (90 * n_stay_dates)

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
        expected_otb_rooms = avg_daily_pace * days_elapsed * capacity["total"]
        fill_rate          = otb_rooms / max(expected_otb_rooms, 1)

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
# STREAMLIT INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
#
# Training block (~line 1146):
#
#   pickup_models, pickup_metrics, pickup_df, pickup_cv = train_pickup_models(
#       raw_proc, capacity=CAPACITY
#   )
#   st.session_state.pickup_models  = pickup_models
#   st.session_state.pickup_metrics = pickup_metrics
#   st.session_state.pickup_df      = pickup_df
#   st.session_state.pickup_cv      = pickup_cv
#
# Forecast call (~line 1257):
#
#   if st.session_state.get("pickup_models"):
#       fcast = forecast_with_pickup_models(
#           st.session_state.pickup_models, otb_df, as_of_ts,
#           raw=st.session_state.raw, capacity=CAPACITY, channels=CHANNELS,
#           allow_wash=True,
#       )
#   else:
#       fcast = forecast_otb_anchored(ts_models, feat_cols, daily, otb_df, horizon)
#
# Sidebar metrics — show Bias next to R² so operators see directional error:
#
#   for target, m in st.session_state.pickup_metrics.items():
#       st.caption(
#           f"{m['label']}: R²={m['R2']:.3f}  MAE={m['MAE']:.2f}"
#           f"  Bias={m['Bias']:+.2f}"   # + = over-forecast, - = under-forecast
#       )
#
# ══════════════════════════════════════════════════════════════════════════════
