"""Create the MongoDB collections/indexes and load the hourly dataset.

Usage (repository root):
    python -m database.scripts.load_mongo            # drop + recreate + load
    python -m database.scripts.load_mongo --limit 500

Collections
  households        reference document for the metered household
  hourly_readings   one document per hour (measurements / sub_metering_wh / quality / calendar sub-documents)
  daily_summaries   pre-aggregated per-day statistics ("computed pattern") for fast dashboards
  models            registry of trained models (hyper-parameters & metrics embedded)
  forecasts         predictions written by the prediction script

Timestamps: the source has no time zone. Naive datetimes are stored as-is (MongoDB treats them as UTC),
so what you read back is exactly the original local clock time.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient

from database import config as C

SUB_FIELDS = {"Sub_metering_1": "kitchen", "Sub_metering_2": "laundry_room",
              "Sub_metering_3": "water_heater_ac", "Other_energy_Wh": "other"}


def nan_to_none(v):
    return None if pd.isna(v) else float(v)


def reading_doc(r, hh=C.DEFAULT_HOUSEHOLD_CODE) -> dict:
    ts = r.timestamp.to_pydatetime()
    return {
        "household_id": hh,
        "timestamp": ts,
        "measurements": {
            "global_active_power_kw": nan_to_none(r.Global_active_power),
            "global_reactive_power_kvar": nan_to_none(r.Global_reactive_power),
            "voltage_v": nan_to_none(r.Voltage),
            "global_intensity_a": nan_to_none(r.Global_intensity),
        },
        "sub_metering_wh": {name: nan_to_none(getattr(r, col)) for col, name in SUB_FIELDS.items()},
        "quality": {"n_obs_minutes": int(r.n_obs), "is_valid_hour": bool(r.is_valid_hour)},
        "calendar": {"year": ts.year, "month": ts.month, "day": ts.day, "hour": ts.hour,
                     "weekday": ts.weekday(), "is_weekend": ts.weekday() >= 5},
    }


def daily_summaries(df: pd.DataFrame, hh=C.DEFAULT_HOUSEHOLD_CODE) -> list[dict]:
    v = df[df.is_valid_hour].copy()
    v["date"] = v.timestamp.dt.floor("D")
    out = []
    for date, g in v.groupby("date"):
        peak = g.loc[g.Global_active_power.idxmax()]
        out.append({
            "household_id": hh, "date": date.to_pydatetime(), "valid_hours": int(len(g)),
            "energy_kwh": round(float(g.Global_active_power.sum()), 4),
            "mean_kw": round(float(g.Global_active_power.mean()), 4),
            "max_kw": round(float(peak.Global_active_power), 4),
            "peak_hour": int(peak.timestamp.hour),
            "min_kw": round(float(g.Global_active_power.min()), 4),
            "sub_metering_kwh": {name: round(float(g[col].sum()) / 1000, 4) for col, name in SUB_FIELDS.items()},
        })
    return out


NUM = ["double", "int", "null"]
READING_VALIDATOR = {"$jsonSchema": {
    "bsonType": "object",
    "required": ["household_id", "timestamp", "measurements", "sub_metering_wh", "quality", "calendar"],
    "properties": {
        "household_id": {"bsonType": "string"},
        "timestamp": {"bsonType": "date"},
        "measurements": {"bsonType": "object", "properties": {
            "global_active_power_kw": {"bsonType": NUM, "minimum": 0},
            "global_reactive_power_kvar": {"bsonType": NUM, "minimum": 0},
            "voltage_v": {"bsonType": NUM, "minimum": 0},
            "global_intensity_a": {"bsonType": NUM, "minimum": 0}}},
        "sub_metering_wh": {"bsonType": "object"},
        "quality": {"bsonType": "object", "required": ["n_obs_minutes", "is_valid_hour"], "properties": {
            "n_obs_minutes": {"bsonType": "int", "minimum": 0, "maximum": 60},
            "is_valid_hour": {"bsonType": "bool"}}},
    }}}
FORECAST_VALIDATOR = {"$jsonSchema": {
    "bsonType": "object",
    "required": ["household_id", "model", "made_at", "target_timestamp", "predicted_kw"],
    "properties": {"household_id": {"bsonType": "string"}, "made_at": {"bsonType": "date"},
                   "target_timestamp": {"bsonType": "date"}, "predicted_kw": {"bsonType": ["double", "int"], "minimum": 0},
                   "actual_kw": {"bsonType": ["double", "int", "null"]}}}}


def main(limit=None):
    t0 = time.time()
    df = pd.read_csv(C.HOURLY_CSV, parse_dates=["timestamp"])
    if limit:
        df = df.head(limit)
    client = MongoClient(C.MONGO_URI)
    client.drop_database(C.MONGO_DB)
    db = client[C.MONGO_DB]
    # schema validation keeps the flexible documents consistent (rejects e.g. negative kW or a missing timestamp)
    db.create_collection("hourly_readings", validator=READING_VALIDATOR)
    db.create_collection("forecasts", validator=FORECAST_VALIDATOR)

    db.households.insert_one({
        "_id": C.DEFAULT_HOUSEHOLD_CODE, "name": "Household 1",
        "description": "Single household whose power was metered every minute (Dec 2006 - Nov 2010)",
        "data_source": "UCI Individual household electric power consumption, hourly aggregate from Task 1",
        "circuits": [
            {"code": "sub_metering_1", "field": "kitchen", "name": "Kitchen", "unit": "Wh"},
            {"code": "sub_metering_2", "field": "laundry_room", "name": "Laundry room", "unit": "Wh"},
            {"code": "sub_metering_3", "field": "water_heater_ac", "name": "Water heater & A/C", "unit": "Wh"},
            {"code": "other", "field": "other", "name": "Other (not sub-metered)", "unit": "Wh", "derived": True},
        ],
    })

    docs = [reading_doc(r) for r in df.itertuples(index=False)]
    for i in range(0, len(docs), 5000):
        db.hourly_readings.insert_many(docs[i:i + 5000], ordered=True)
    print(f"hourly_readings: {len(docs):,} documents")

    daily = daily_summaries(df)
    db.daily_summaries.insert_many(daily)
    print(f"daily_summaries: {len(daily):,} documents")

    db.models.insert_one({
        "experiment_id": "E4", "model_name": "Random Forest (tuned)", "version": "1.0",
        "target": "Global_active_power", "horizon_hours": 1,
        "hyperparameters": {"n_estimators": 50, "min_samples_leaf": 15, "max_features": 0.5, "max_depth": 12},
        "metrics": {"validation": {"mae": 0.3502}, "test": {"mae": 0.3046, "rmse": 0.4474, "r2": 0.6004}},
        "preprocessing": {"min_observed_minutes_per_hour": 45, "causal_fill_limit_hours": 3},
        "split": "chronological 70/15/15", "source": "Task 1 notebook",
    })

    # ---- indexes (the unique compound index also serves 'latest' and 'date range' queries)
    db.hourly_readings.create_index([("household_id", ASCENDING), ("timestamp", ASCENDING)], unique=True, name="uq_household_ts")
    db.hourly_readings.create_index([("calendar.year", ASCENDING), ("calendar.month", ASCENDING)], name="ix_year_month")
    db.daily_summaries.create_index([("household_id", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_household_date")
    db.models.create_index([("experiment_id", ASCENDING), ("version", ASCENDING)], unique=True, name="uq_exp_version")
    db.forecasts.create_index([("household_id", ASCENDING), ("target_timestamp", DESCENDING)], name="ix_household_target")
    print("indexes:", {c: [i for i in db[c].index_information()] for c in ("hourly_readings", "daily_summaries", "models", "forecasts")})
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
