"""Tests for the Task 3 API and the Task 4 prediction script (run: ``pytest tests -v``).

* MongoDB routes run against an in-memory ``mongomock`` database, so no server is needed.
* The end-to-end test drives prediction.predict through the real HTTP routes (API -> preprocess -> model).
* MySQL routes need a running server loaded by ``database.scripts.load_sql``; they are skipped otherwise.
  They work on a far-future hour (2030) and delete it again, so the loaded data is left unchanged.
"""
import sys
import unittest
from datetime import datetime
from pathlib import Path

import mongomock
import pandas as pd
from fastapi.testclient import TestClient
from pymongo import ASCENDING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import db as api_db  # noqa: E402
from api.main import app  # noqa: E402
from database import config as C  # noqa: E402
from database.scripts.load_mongo import reading_doc  # noqa: E402
from prediction import predict as P  # noqa: E402

NEW = {"timestamp": "2030-01-01T10:00:00", "global_active_power_kw": 1.5, "global_reactive_power_kvar": 0.1,
       "voltage_v": 240.0, "global_intensity_a": 6.2, "sub_metering_1_wh": 100.0, "sub_metering_2_wh": 200.0,
       "sub_metering_3_wh": 300.0, "n_obs_minutes": 60}


def seeded_mongo(n_hours: int = 400):
    """mongomock database holding the last ``n_hours`` of the real hourly data, built like load_mongo does."""
    db = mongomock.MongoClient()[C.MONGO_DB]
    df = pd.read_csv(C.HOURLY_CSV, parse_dates=["timestamp"]).tail(n_hours)
    db.hourly_readings.insert_many([reading_doc(r) for r in df.itertuples(index=False)])
    db.hourly_readings.create_index([("household_id", ASCENDING), ("timestamp", ASCENDING)], unique=True)
    db.models.insert_one({"experiment_id": "E4", "model_name": "Random Forest (tuned)", "version": "1.0"})
    return db


class MongoCrudTest(unittest.TestCase):
    def setUp(self):
        self.db = seeded_mongo()
        app.dependency_overrides[api_db.get_mongo_db] = lambda: self.db
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_crud_cycle(self):
        r = self.client.post("/mongo/readings", json=NEW)
        self.assertEqual(r.status_code, 201, r.text)
        created = r.json()
        self.assertTrue(created["is_valid_hour"])
        self.assertAlmostEqual(created["other_energy_wh"], 1500 - 600)       # derived like Task 1
        doc = self.db.hourly_readings.find_one({"timestamp": datetime(2030, 1, 1, 10)})
        self.assertEqual(doc["sub_metering_wh"]["kitchen"], 100.0)           # stored in the Task 2 nested layout
        self.assertEqual(doc["calendar"]["weekday"], 1)

        self.assertEqual(self.client.post("/mongo/readings", json=NEW).status_code, 409)   # same hour twice

        rid = created["id"]
        self.assertEqual(self.client.get(f"/mongo/readings/{rid}").json()["voltage_v"], 240.0)
        upd = self.client.put(f"/mongo/readings/{rid}", json={"global_active_power_kw": 2.0, "n_obs_minutes": 30}).json()
        self.assertEqual(upd["global_active_power_kw"], 2.0)
        self.assertAlmostEqual(upd["other_energy_wh"], 2000 - 600)            # recomputed after the update
        self.assertFalse(upd["is_valid_hour"])                                # 30 < 45 observed minutes

        self.assertEqual(self.client.delete(f"/mongo/readings/{rid}").status_code, 200)
        self.assertEqual(self.client.get(f"/mongo/readings/{rid}").status_code, 404)
        self.assertEqual(self.client.delete(f"/mongo/readings/{rid}").status_code, 404)
        self.assertEqual(self.client.get("/mongo/readings/not-an-id").status_code, 404)

    def test_latest_and_range(self):
        latest = self.client.get("/mongo/readings/latest").json()
        self.assertEqual(latest["timestamp"], "2010-11-26T21:00:00")
        self.assertIsNone(latest["global_active_power_kw"])                   # final hour is incomplete
        valid = self.client.get("/mongo/readings/latest", params={"valid_only": True}).json()
        self.assertEqual(valid["timestamp"], "2010-11-26T20:00:00")

        rows = self.client.get("/mongo/readings/range",
                               params={"start": "2010-11-26T00:00:00", "end": "2010-11-26T05:00:00"}).json()
        self.assertEqual([r["timestamp"][11:13] for r in rows], ["00", "01", "02", "03", "04", "05"])  # inclusive, sorted
        bad = self.client.get("/mongo/readings/range", params={"start": "2010-11-26T05:00:00", "end": "2010-11-26T00:00:00"})
        self.assertEqual(bad.status_code, 400)

    def test_validation(self):
        self.assertEqual(self.client.post("/mongo/readings", json={**NEW, "timestamp": "2030-01-01T10:30:00"}).status_code, 422)
        self.assertEqual(self.client.post("/mongo/readings", json={**NEW, "voltage_v": -1}).status_code, 422)
        too_much = {**NEW, "global_active_power_kw": 0.1}                    # 100 Wh total < 600 Wh sub-metered
        self.assertEqual(self.client.post("/mongo/readings", json=too_much).status_code, 422)


class EndToEndPredictionTest(unittest.TestCase):
    """Task 4 against the Task 3 routes: fetch over HTTP -> preprocess -> load model -> predict -> store."""

    def setUp(self):
        self.db = seeded_mongo()
        app.dependency_overrides[api_db.get_mongo_db] = lambda: self.db
        self.api = P.ApiClient("http://testserver", "mongo", session=TestClient(app))

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_forecast_matches_task1_and_is_saved(self):
        model, meta = P.load_model()
        ref = pd.read_csv(P.ROOT / "outputs" / "tables" / "final_test_predictions.csv", parse_dates=["timestamp"])
        target = ref.iloc[-50]
        result = P.forecast(self.api, (target.timestamp - pd.Timedelta(hours=1)).to_pydatetime(), model, meta)
        # same prediction as the Task 1 notebook produced for this hour (table is rounded to 5 decimals)
        self.assertAlmostEqual(result["predicted_kw"], target.prediction_kW, places=3)
        self.assertAlmostEqual(result["actual_kw"], target.actual_kW, places=3)

        latest = P.forecast(self.api, None, model, meta)                      # default origin = latest valid hour
        self.assertEqual(latest["target_timestamp"], datetime(2010, 11, 26, 21))
        self.assertIsNone(latest["actual_kw"])
        saved = self.api.save_forecast({"made_at": latest["made_at"].isoformat(),
                                        "target_timestamp": latest["target_timestamp"].isoformat(),
                                        "predicted_kw": latest["predicted_kw"]})
        self.assertEqual(saved["experiment_id"], "E4")
        self.api.save_forecast({"made_at": latest["made_at"].isoformat(), "predicted_kw": 9.9,
                                "target_timestamp": latest["target_timestamp"].isoformat()})
        self.assertEqual(self.db.forecasts.count_documents({}), 1)            # re-forecast replaces, not duplicates

    def test_gap_in_history_is_reported(self):
        self.db.hourly_readings.delete_many({"timestamp": {"$gte": datetime(2010, 11, 26, 10),
                                                           "$lte": datetime(2010, 11, 26, 15)}})
        with self.assertRaisesRegex(ValueError, "fill limit"):
            P.forecast(self.api, datetime(2010, 11, 26, 20))


def mysql_available() -> bool:
    try:
        for conn in api_db.get_sql_conn():
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS n FROM ml_models")
                return cur.fetchone()["n"] > 0
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(mysql_available(), "MySQL power_db not reachable (run database.scripts.load_sql first)")
class SqlCrudTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        for conn in api_db.get_sql_conn():   # clean up after an interrupted earlier run
            with conn.cursor() as cur:
                cur.execute("DELETE FROM hourly_readings WHERE reading_ts >= '2030-01-01'")
            conn.commit()

    def test_crud_cycle(self):
        r = self.client.post("/sql/readings", json=NEW)
        self.assertEqual(r.status_code, 201, r.text)
        created = r.json()
        self.assertAlmostEqual(created["other_energy_wh"], 900.0)
        self.assertEqual(self.client.post("/sql/readings", json=NEW).status_code, 409)
        rid = created["id"]
        upd = self.client.put(f"/sql/readings/{rid}", json={"sub_metering_3_wh": 500.0}).json()
        self.assertAlmostEqual(upd["other_energy_wh"], 700.0)
        self.assertEqual(self.client.get("/sql/readings/latest").json()["id"], rid)
        self.assertEqual(self.client.delete(f"/sql/readings/{rid}").status_code, 200)
        self.assertEqual(self.client.get(f"/sql/readings/{rid}").status_code, 404)

    def test_latest_and_range(self):
        self.assertEqual(self.client.get("/sql/readings/latest", params={"valid_only": True}).json()["timestamp"],
                         "2010-11-26T20:00:00")
        rows = self.client.get("/sql/readings/range",
                               params={"start": "2010-11-26T00:00:00", "end": "2010-11-26T05:00:00"}).json()
        self.assertEqual(len(rows), 6)
        self.assertIsNotNone(rows[0]["sub_metering_3_wh"])


if __name__ == "__main__":
    unittest.main()
