"""Leakage-safe feature engineering for h-step-ahead hourly forecasting.

Convention used everywhere
--------------------------
Every row of the supervised frame is indexed by the **target timestamp T** (the hour being predicted).
The forecast origin is T - h; only data with timestamp <= T - h may be used as input.

* ``<col>_lag<k>``      = value of ``col`` at T - h - (k - 1)  -> lag1 is the most recent value known at the origin.
* ``gap_roll_mean_<w>h``/``gap_roll_std_<w>h`` = statistics of the w values ending at T - h (shifted before rolling).
* Calendar features are derived from T itself (the timestamp being predicted), which is known in advance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Global_active_power"
TARGET_ALIAS = "gap"
DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 48, 168)
DEFAULT_WINDOWS = (3, 6, 24, 168)
DEFAULT_EXOG = ("Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1",
                "Sub_metering_2", "Sub_metering_3", "Other_energy_Wh")


def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features of the (target) timestamps, including cyclical encodings."""
    cal = pd.DataFrame(index=index)
    cal["hour"] = index.hour
    cal["day_of_week"] = index.dayofweek
    cal["is_weekend"] = (index.dayofweek >= 5).astype(int)
    cal["month"] = index.month
    cal["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    cal["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    cal["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    cal["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
    return cal


def make_lag_features(series: pd.Series, lags, horizon: int = 1, name: str = TARGET_ALIAS) -> pd.DataFrame:
    """Lag features available at the forecast origin (lag1 = latest known value)."""
    out = {}
    for k in lags:
        if k < 1:
            raise ValueError("lags must be >= 1 (lag 0 would be the target itself)")
        out[f"{name}_lag{k}"] = series.shift(k + horizon - 1)
    return pd.DataFrame(out, index=series.index)


def make_rolling_features(series: pd.Series, windows, horizon: int = 1, name: str = TARGET_ALIAS) -> pd.DataFrame:
    """Rolling mean / std over windows that end at the forecast origin (series shifted *before* rolling)."""
    past = series.shift(horizon)
    out = {}
    for w in windows:
        roll = past.rolling(window=w, min_periods=w)
        out[f"{name}_roll_mean_{w}h"] = roll.mean()
        if w >= 3:
            out[f"{name}_roll_std_{w}h"] = roll.std()
    return pd.DataFrame(out, index=series.index)


def build_supervised_frame(filled: pd.DataFrame, observed_target: pd.Series, *, horizon: int = 1,
                           lags=DEFAULT_LAGS, windows=DEFAULT_WINDOWS, exog_cols=DEFAULT_EXOG,
                           exog_lags=(1,), target_col: str = TARGET, include_calendar: bool = True):
    """Create (X, y, info) for h-step-ahead forecasting.

    ``filled`` supplies the *inputs* (short gaps may have been causally forward-filled), whereas
    ``observed_target`` supplies the *outcome* y(T) and is never imputed. Rows with any missing
    input or missing outcome are dropped, and the number dropped is reported.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not filled.index.equals(observed_target.index):
        raise ValueError("filled and observed_target must share the same index")
    parts = [make_lag_features(filled[target_col], lags, horizon),
             make_rolling_features(filled[target_col], windows, horizon)]
    for col in exog_cols:
        parts.append(make_lag_features(filled[col], exog_lags, horizon, name=col))
    if include_calendar:
        parts.append(add_calendar_features(filled.index))
    X = pd.concat(parts, axis=1)
    y = observed_target.rename(f"{target_col}_t_plus_{horizon}h")
    keep = X.notna().all(axis=1) & y.notna()
    info = {"rows_before_dropping": int(len(X)), "rows_missing_target": int(y.isna().sum()),
            "rows_missing_inputs": int(X.isna().any(axis=1).sum()), "rows_kept": int(keep.sum())}
    return X.loc[keep], y.loc[keep], info


def lag_correlations(series: pd.Series, lags) -> pd.DataFrame:
    """Pearson correlation between y(t) and y(t-k), pairwise-complete (past values only)."""
    rows = [{"lag_hours": k, "correlation": series.corr(series.shift(k))} for k in lags]
    return pd.DataFrame(rows)


def partial_autocorrelation(acf: np.ndarray) -> np.ndarray:
    """Partial autocorrelations via the Durbin-Levinson recursion. ``acf[0]`` must be 1 (lag 0)."""
    n = len(acf) - 1
    pacf = np.zeros(n + 1)
    pacf[0] = 1.0
    phi_prev = np.zeros(n + 1)
    for k in range(1, n + 1):
        if k == 1:
            phi_kk = acf[1]
            phi = np.zeros(n + 1)
            phi[1] = phi_kk
        else:
            num = acf[k] - np.dot(phi_prev[1:k], acf[1:k][::-1])
            den = 1.0 - np.dot(phi_prev[1:k], acf[1:k])
            phi_kk = num / den
            phi = phi_prev.copy()
            phi[1:k] = phi_prev[1:k] - phi_kk * phi_prev[1:k][::-1]
            phi[k] = phi_kk
        pacf[k] = phi_kk
        phi_prev = phi
    return pacf


def leakage_check(filled: pd.DataFrame, observed_target: pd.Series, *, horizon: int = 1, n_trials: int = 5,
                  seed: int = 42, **kwargs) -> bool:
    """Programmatic leakage test.

    Corrupt all data from a random time t0 onwards, rebuild the features and verify that every feature row
    whose target timestamp T satisfies T - horizon < t0 (all inputs strictly before t0) is unchanged.
    """
    rng = np.random.default_rng(seed)
    X_ref, _, _ = build_supervised_frame(filled, observed_target, horizon=horizon, **kwargs)
    for _ in range(n_trials):
        t0 = filled.index[int(rng.integers(len(filled) // 4, 3 * len(filled) // 4))]
        corrupted = filled.copy()
        num = corrupted.select_dtypes("number").columns
        corrupted.loc[corrupted.index >= t0, num] += 1e6
        X_c, _, _ = build_supervised_frame(corrupted, observed_target, horizon=horizon, **kwargs)
        safe_rows = X_ref.index[X_ref.index < t0 + pd.Timedelta(hours=horizon)]
        common = safe_rows.intersection(X_c.index)
        # calendar features are deterministic functions of T so they never change either
        if not np.allclose(X_ref.loc[common].to_numpy(float), X_c.loc[common].to_numpy(float)):
            return False
    return True


def feature_dictionary(columns, horizon: int = 1, lags=DEFAULT_LAGS, windows=DEFAULT_WINDOWS) -> pd.DataFrame:
    """Human-readable documentation for each engineered feature."""
    rows = []
    for c in columns:
        if c.startswith("gap_lag"):
            k = int(c.replace("gap_lag", ""))
            rows.append((c, f"Global_active_power {k} hour(s) before the forecast origin (lag1 = last known hour)",
                         f"shift({k + horizon - 1}) of hourly mean power", "autoregressive memory; k=24/168 capture daily/weekly cycle"))
        elif "_roll_mean_" in c:
            w = c.split("_")[-1]
            rows.append((c, f"Mean of the last {w} hourly target values up to the origin",
                         f"shift({horizon}).rolling({w[:-1]}).mean()", "smoothed recent level; reduces hourly noise"))
        elif "_roll_std_" in c:
            w = c.split("_")[-1]
            rows.append((c, f"Std. deviation of the last {w} hourly target values up to the origin",
                         f"shift({horizon}).rolling({w[:-1]}).std()", "recent volatility"))
        elif c.endswith("_lag1"):
            rows.append((c, f"Last known hourly value of {c[:-5]}", f"shift({horizon})", "co-measured appliance / electrical state"))
        elif c in ("hour", "day_of_week", "month", "is_weekend"):
            desc = {"hour": "hour of day (0-23) of the predicted hour", "day_of_week": "Monday=0 ... Sunday=6 of the predicted hour",
                    "month": "calendar month of the predicted hour", "is_weekend": "1 if predicted hour is Saturday/Sunday"}[c]
            rows.append((c, desc, f"from target timestamp T (.{ 'hour' if c=='hour' else 'dayofweek' if c in ('day_of_week','is_weekend') else 'month'})",
                         "calendar seasonality (known in advance)"))
        else:
            rows.append((c, "Cyclical sin/cos encoding of hour or weekday", "sin/cos(2*pi*x/period)", "lets linear models see cyclic continuity"))
    return pd.DataFrame(rows, columns=["feature", "definition", "calculation", "forecasting_purpose"])
