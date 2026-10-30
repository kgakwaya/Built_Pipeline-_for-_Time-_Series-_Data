"""Write database/mongo/sample_documents.json from the live database (one example per collection).

Usage: python -m database.scripts.export_sample_documents
"""
from __future__ import annotations

import json
from datetime import datetime

from bson import json_util
from pymongo import MongoClient

from database import config as C


def main():
    db = MongoClient(C.MONGO_URI)[C.MONGO_DB]
    hh = C.DEFAULT_HOUSEHOLD_CODE
    valid = db.hourly_readings.find_one({"household_id": hh, "quality.is_valid_hour": True}, sort=[("timestamp", 1)])
    invalid = db.hourly_readings.find_one({"household_id": hh, "quality.is_valid_hour": False}, sort=[("timestamp", 1)])
    samples = {
        "households": db.households.find_one({"_id": hh}),
        "hourly_readings (valid hour)": valid,
        "hourly_readings (invalid hour: values are null, quality flags explain why)": invalid,
        "daily_summaries": db.daily_summaries.find_one({"household_id": hh, "valid_hours": 24}, sort=[("date", 1)]),
        "models": db.models.find_one({"experiment_id": "E4"}),
        "forecasts (illustrative document: values invented, shows the shape the prediction script writes)": {
            "household_id": hh, "model": {"experiment_id": "E4", "name": "Random Forest (tuned)", "version": "1.0"},
            "made_at": datetime(2010, 11, 26, 20), "target_timestamp": datetime(2010, 11, 26, 21), "horizon_hours": 1,
            "predicted_kw": 1.2873, "actual_kw": None, "created_at": datetime(2026, 10, 29, 9, 30)},
    }
    out = C.ROOT / "database" / "mongo" / "sample_documents.json"
    out.write_text(json_util.dumps(samples, indent=2, json_options=json_util.RELAXED_JSON_OPTIONS), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
