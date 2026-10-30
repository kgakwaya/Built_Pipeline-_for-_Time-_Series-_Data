"""Task 4: one-hour-ahead forecast of household active power, end to end.

    1. fetch    - GET the latest valid reading and the preceding history from the Task 3 API
    2. preprocess - same pipeline as Task 1 (src/preprocessing.py, src/feature_engineering.py):
                  regular hourly grid -> 45-minute quality rule -> causal forward fill (<= 3 h)
                  -> lag / rolling-window / exogenous-lag / calendar features
    3. load     - the tuned Random Forest saved by Task 1 (outputs/models/best_model.joblib)
    4. predict  - forecast Global_active_power for origin + 1 h; optionally POST it to /forecasts

Run from the repository root while the API is up (``uvicorn api.main:app``):
    python -m prediction.predict                          # MySQL backend, latest valid hour as origin
    python -m prediction.predict --backend mongo --save   # MongoDB, and store the forecast
    python -m prediction.predict --origin 2010-06-15T18:00:00
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import feature_engineering as fe, preprocessing as pp  # noqa: E402

MODEL_PATH = ROOT / "outputs" / "models" / "best_model.joblib"
META_PATH = ROOT / "outputs" / "models" / "best_model_metadata.json"
# API field names -> column names used by Task 1 (hourly_observed.csv)
API_TO_TASK1 = {
    "global_active_power_kw": "Global_active_power", "global_reactive_power_kvar": "Global_reactive_power",
    "voltage_v": "Voltage", "global_intensity_a": "Global_intensity",
    "sub_metering_1_wh": "Sub_metering_1", "sub_metering_2_wh": "Sub_metering_2", "sub_metering_3_wh": "Sub_metering_3",
    "other_energy_wh": "Other_energy_Wh", "n_obs_minutes": "n_obs",
}
VALUE_COLS = [c for c in API_TO_TASK1.values() if c != "n_obs"]


# ----------------------------------------------------------------------------- 1. fetch
class ApiClient:
    def __init__(self, base_url: str, backend: str, timeout: float = 30, session=None):
        self.base = f"{base_url.rstrip('/')}/{backend}"
        self.timeout = timeout
        self.http = session or requests.Session()

    def _get(self, path: str, **params):
        resp = self.http.get(f"{self.base}{path}", params=params, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"GET {self.base}{path} -> HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def latest_valid(self) -> dict:
        return self._get("/readings/latest", valid_only="true")

    def range(self, start: datetime, end: datetime) -> list[dict]:
        return self._get("/readings/range", start=start.isoformat(), end=end.isoformat())

    def save_forecast(self, payload: dict) -> dict:
        resp = self.http.post(f"{self.base}/forecasts", json=payload, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"POST {self.base}/forecasts -> HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()


def history_hours_needed(meta: dict) -> int:
    """Longest look-back of any feature plus room for the causal fill to reach back over a short gap."""
    lags = [int(f.rsplit("lag", 1)[1]) for f in meta["features"] if "_lag" in f]
    windows = [int(f.split("_")[-1].rstrip("h")) for f in meta["features"] if "_roll_" in f]
    return max(lags + windows) + meta["preprocessing"]["causal_fill_limit_hours"] + 24


def fetch_history(client: ApiClient, hours: int, origin: datetime | None = None) -> tuple[pd.DataFrame, datetime]:
    """Return the raw readings for (origin - hours, origin] and the forecast origin itself."""
    if origin is None:
        origin = datetime.fromisoformat(client.latest_valid()["timestamp"])
    records = client.range(origin - timedelta(hours=hours - 1), origin)
    if not records:
        raise RuntimeError(f"the API returned no readings up to {origin}")
    return pd.DataFrame(records), origin


# ----------------------------------------------------------------------------- 2. preprocess
def to_hourly_frame(records: pd.DataFrame, origin: datetime, hours: int, min_obs: int) -> pd.DataFrame:
    """API records -> the Task 1 hourly layout on a regular grid, with the quality rule applied."""
    df = records.rename(columns=API_TO_TASK1)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    grid = pd.date_range(pd.Timestamp(origin) - pd.Timedelta(hours=hours - 1), pd.Timestamp(origin), freq="h",
                         name="timestamp")
    hourly = df.reindex(grid)[[*VALUE_COLS, "n_obs"]].astype(float)   # hours absent from the database become NaN
    hourly["n_obs"] = hourly["n_obs"].fillna(0)
    hourly["is_valid_hour"] = hourly["n_obs"] >= min_obs
    hourly.loc[~hourly["is_valid_hour"], VALUE_COLS] = np.nan            # Task 1: incomplete hours are missing
    subs = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    # same derivation as pp.aggregate_to_hourly (recomputed so a stale stored value cannot leak in)
    hourly["Other_energy_Wh"] = hourly["Global_active_power"] * 1000.0 - hourly[subs].sum(axis=1, skipna=False)
    pp.validate_chronological(hourly.index)
    return hourly


def build_features(hourly: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, dict]:
    """Causal fill + feature engineering for the single target hour T = origin + horizon."""
    h = int(meta["horizon_hours"])
    filled, flags = pp.causal_fill(hourly, VALUE_COLS, meta["preprocessing"]["causal_fill_limit_hours"])
    # append the hour(s) being predicted: the feature functions index rows by the target timestamp T
    future = pd.date_range(filled.index[-1] + pd.Timedelta(hours=1), periods=h, freq="h", name="timestamp")
    ext = pd.concat([filled, pd.DataFrame(index=future, columns=filled.columns, dtype=float)])
    parts = [fe.make_lag_features(ext[fe.TARGET], fe.DEFAULT_LAGS, h),
             fe.make_rolling_features(ext[fe.TARGET], fe.DEFAULT_WINDOWS, h),
             *(fe.make_lag_features(ext[c], (1,), h, name=c) for c in fe.DEFAULT_EXOG),
             fe.add_calendar_features(ext.index)]
    X = pd.concat(parts, axis=1).iloc[[-1]][meta["features"]]   # exact column order the model was trained on
    missing = X.columns[X.isna().iloc[0]].tolist()
    if missing:
        raise ValueError(f"cannot forecast {X.index[0]}: history has gaps longer than the "
                         f"{meta['preprocessing']['causal_fill_limit_hours']} h fill limit; missing features: {missing}")
    info = {"hours_in_window": len(hourly), "hours_missing_or_invalid": int(hourly[fe.TARGET].isna().sum()),
            "target_hours_imputed": int(flags[fe.TARGET].sum())}
    return X, info


# ----------------------------------------------------------------------------- 3 + 4. load and predict
def load_model(model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return joblib.load(model_path), meta


def forecast(client: ApiClient, origin: datetime | None = None, model=None, meta=None) -> dict:
    if model is None:
        model, meta = load_model()
    hours = history_hours_needed(meta)
    records, origin = fetch_history(client, hours, origin)
    hourly = to_hourly_frame(records, origin, hours, meta["preprocessing"]["min_observed_minutes_per_hour"])
    X, info = build_features(hourly, meta)
    target_ts = X.index[0].to_pydatetime()
    predicted = max(float(model.predict(X)[0]), 0.0)   # power cannot be negative
    result = {"model": f"{meta['selected_experiment']} - {meta['model']}", "made_at": origin, "target_timestamp": target_ts,
              "predicted_kw": round(predicted, 4), "persistence_kw": round(float(X["gap_lag1"].iloc[0]), 4),
              "actual_kw": None, **info, "features": X.iloc[0].round(4).to_dict()}
    # the dataset is historical, so the true value is often already stored: report it for comparison
    actual = client.range(target_ts, target_ts)
    if actual and actual[0].get("is_valid_hour") and actual[0].get("global_active_power_kw") is not None:
        result["actual_kw"] = round(actual[0]["global_active_power_kw"], 4)
        result["abs_error_kw"] = round(abs(result["actual_kw"] - result["predicted_kw"]), 4)
    return result


def print_report(r: dict, backend: str, show_features: bool) -> None:
    print(f"\n=== One-hour-ahead forecast ({backend} backend) ===")
    print(f"model              : {r['model']}")
    print(f"forecast origin    : {r['made_at']}   (latest hour used)")
    print(f"target hour        : {r['target_timestamp']}")
    print(f"history window     : {r['hours_in_window']} h, {r['hours_missing_or_invalid']} missing/invalid, "
          f"{r['target_hours_imputed']} target hours forward-filled")
    print(f"PREDICTED power    : {r['predicted_kw']:.4f} kW")
    print(f"persistence (t-1)  : {r['persistence_kw']:.4f} kW")
    if r["actual_kw"] is not None:
        print(f"actual (stored)    : {r['actual_kw']:.4f} kW   |error| = {r['abs_error_kw']:.4f} kW")
    else:
        print("actual             : not available (hour missing, invalid or in the future)")
    if show_features:
        print("\nmodel inputs:")
        for k, v in r["features"].items():
            print(f"  {k:<28} {v}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", default="http://127.0.0.1:8000")
    ap.add_argument("--backend", choices=["sql", "mongo"], default="sql")
    ap.add_argument("--origin", type=datetime.fromisoformat, default=None,
                    help="forecast origin (last observed hour), e.g. 2010-06-15T18:00:00; default: latest valid hour")
    ap.add_argument("--save", action="store_true", help="store the forecast through POST /{backend}/forecasts")
    ap.add_argument("--show-features", action="store_true", help="print the 31 model inputs")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)

    client = ApiClient(args.api_url, args.backend)
    try:
        model, meta = load_model()
        result = forecast(client, args.origin, model, meta)
    except (requests.ConnectionError, requests.Timeout) as exc:
        print(f"error: cannot reach the API at {args.api_url} ({exc.__class__.__name__}). "
              "Start it with: uvicorn api.main:app", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.save:
        saved = client.save_forecast({
            "made_at": result["made_at"].isoformat(), "target_timestamp": result["target_timestamp"].isoformat(),
            "predicted_kw": result["predicted_kw"], "actual_kw": result["actual_kw"],
            "experiment_id": meta["selected_experiment"], "model_version": "1.0"})
        result["saved_forecast_id"] = saved["id"]

    if args.json:
        print(json.dumps(result, default=str, indent=2))
    else:
        print_report(result, args.backend, args.show_features)
        if args.save:
            print(f"saved forecast     : id {result['saved_forecast_id']} in {args.backend} 'forecasts'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
