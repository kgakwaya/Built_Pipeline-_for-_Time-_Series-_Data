"""Timestamp validation, duplicate handling, missing-data analysis and hourly aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

POWER_COLS = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]
SUB_METER_COLS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
# Sub-metering values are watt-hours of active energy per minute (UCI documentation).
MINUTES_PER_HOUR = 60


# ----------------------------------------------------------------------------- timestamps
def inspect_timestamps(index: pd.DatetimeIndex, expected_freq: str = "min") -> dict:
    """Validate ordering, duplicates, sampling interval and grid completeness of a DatetimeIndex."""
    diffs = pd.Series(index[1:] - index[:-1])
    mode_interval = diffs.mode().iloc[0] if len(diffs) else pd.NaT
    expected = pd.date_range(index.min(), index.max(), freq=expected_freq)
    missing_grid = expected.difference(index.unique())
    return {
        "n_rows": int(len(index)),
        "n_unique": int(index.nunique()),
        "is_monotonic_increasing": bool(index.is_monotonic_increasing),
        "duplicate_timestamps": int(index.duplicated().sum()),
        "start": index.min(), "end": index.max(), "span": index.max() - index.min(),
        "median_interval": diffs.median() if len(diffs) else pd.NaT,
        "modal_interval": mode_interval,
        "share_modal_interval": float((diffs == mode_interval).mean()) if len(diffs) else np.nan,
        "n_distinct_intervals": int(diffs.nunique()),
        "expected_grid_points": int(len(expected)),
        "timestamps_missing_from_grid": int(len(missing_grid)),
    }


def handle_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Documented duplicate policy (the index must be a DatetimeIndex).

    1. Fully identical rows (same timestamp *and* values) are dropped, keeping the first.
    2. Remaining duplicate timestamps with conflicting values are averaged (numeric mean ignores NaN).
    The returned report quantifies each step so nothing is dropped silently.
    """
    n0 = len(df)
    dup_ts = int(df.index.duplicated().sum())
    dup_mask = df.reset_index().duplicated().values
    dup_rows = int(dup_mask.sum())
    out = df[~dup_mask]
    conflicting = int(out.index.duplicated().sum())
    if conflicting:
        out = out.groupby(level=0).mean()
    out = out.sort_index()
    return out, {"rows_in": n0, "duplicate_timestamps": dup_ts, "duplicate_rows": dup_rows,
                 "conflicting_timestamps_averaged": conflicting, "rows_out": len(out)}


def validate_chronological(index: pd.Index) -> None:
    """Raise if the index is not strictly increasing (required for lags, rolling windows, splits)."""
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise AssertionError("Index must be strictly increasing and unique.")


# ----------------------------------------------------------------------------- missing data
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count / percentage per column plus a TOTAL row."""
    n = len(df)
    rep = pd.DataFrame({"missing_count": df.isna().sum(), "missing_pct": df.isna().mean() * 100})
    rep.loc["ALL_CELLS"] = [int(df.isna().sum().sum()), df.isna().sum().sum() / (n * df.shape[1]) * 100]
    return rep


def find_runs(mask: pd.Series) -> pd.DataFrame:
    """Return start, end and length of each consecutive True-run of a boolean Series with a time index."""
    mask = mask.astype(bool)
    group = (mask != mask.shift()).cumsum()
    rows = []
    for _, chunk in mask.groupby(group):
        if chunk.iloc[0]:
            rows.append({"start": chunk.index[0], "end": chunk.index[-1], "length": int(len(chunk))})
    return pd.DataFrame(rows, columns=["start", "end", "length"])


# ----------------------------------------------------------------------------- aggregation
def aggregate_to_hourly(df: pd.DataFrame, min_obs: int = 45) -> pd.DataFrame:
    """Aggregate a regular 1-minute frame to hourly resolution.

    Rules (documented in the notebook table):
      * Global_active_power, Global_reactive_power (kW), Voltage (V), Global_intensity (A): mean of the
        observed minutes -> hourly mean power / voltage / current.
      * Sub_metering_1..3 (Wh per minute): mean of observed minutes x 60 -> energy in Wh during the hour.
        (Scaling by 60 extrapolates incomplete hours linearly; hours below ``min_obs`` are invalid anyway.)
      * Other_energy_Wh = Global_active_power_mean*1000 - sum(sub-metering Wh): active energy not captured
        by the three sub-meters (UCI documentation formula, hourly version).
    Hours with fewer than ``min_obs`` observed minutes are set to NaN (never filled here). Columns
    ``n_obs`` (observed minutes) and ``is_valid_hour`` document this.
    """
    validate_chronological(df.index)
    grouped = df.resample("h")
    n_obs = grouped["Global_active_power"].count().rename("n_obs")
    hourly = grouped[POWER_COLS].mean()
    for col in SUB_METER_COLS:
        hourly[col] = grouped[col].mean() * MINUTES_PER_HOUR
    hourly["Other_energy_Wh"] = hourly["Global_active_power"] * 1000.0 - hourly[SUB_METER_COLS].sum(axis=1, skipna=False)
    hourly["n_obs"] = n_obs
    hourly["is_valid_hour"] = n_obs >= min_obs
    value_cols = [c for c in hourly.columns if c not in ("n_obs", "is_valid_hour")]
    hourly.loc[~hourly["is_valid_hour"], value_cols] = np.nan
    hourly.index.name = "timestamp"
    return hourly


# ----------------------------------------------------------------------------- imputation
def causal_fill(df: pd.DataFrame, cols: list[str], limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-fill at most ``limit`` consecutive missing steps (uses past values only).

    Returns the filled copy and a boolean frame flagging imputed cells. Longer gaps stay NaN.
    """
    filled = df.copy()
    flags = pd.DataFrame(False, index=df.index, columns=cols)
    for col in cols:
        before = df[col].isna()
        filled[col] = df[col].ffill(limit=limit)
        flags[col] = before & filled[col].notna()
    return filled, flags


def imputation_experiment(series: pd.Series, gap_lengths=(1, 2, 3), n_gaps: int = 300, seed: int = 42) -> pd.DataFrame:
    """Compare imputation strategies by masking *known* values and measuring recovery error.

    Strategies: causal forward fill (past only), time-based linear interpolation (needs the future value),
    and same-hour-previous-day (past only). Only windows whose context is fully observed are used.
    """
    rng = np.random.default_rng(seed)
    obs = series.copy()
    valid = obs.notna().to_numpy()
    rows = []
    for L in gap_lengths:
        errs = {"ffill (causal)": [], "linear interpolation (uses future)": [], "same hour yesterday (causal)": []}
        tries = 0
        while len(errs["ffill (causal)"]) < n_gaps and tries < n_gaps * 50:
            tries += 1
            s = int(rng.integers(25, len(obs) - L - 2))
            need = valid[s - 25:s + L + 1]       # day before, gap, and the right anchor must be observed
            if not need.all():
                continue
            true = obs.iloc[s:s + L].to_numpy()
            left, right = obs.iloc[s - 1], obs.iloc[s + L]
            ff = np.full(L, left)
            lin = left + (right - left) * (np.arange(1, L + 1) / (L + 1))
            yday = obs.iloc[s - 24:s + L - 24].to_numpy()
            errs["ffill (causal)"].append(true - ff)
            errs["linear interpolation (uses future)"].append(true - lin)
            errs["same hour yesterday (causal)"].append(true - yday)
        for name, e in errs.items():
            e = np.concatenate(e)
            rows.append({"gap_length_h": L, "method": name, "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
                         "n_values": len(e)})
    return pd.DataFrame(rows)
