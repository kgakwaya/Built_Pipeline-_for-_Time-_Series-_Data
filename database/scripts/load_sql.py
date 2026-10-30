"""Create the MySQL schema and load the hourly dataset.

Usage (from the repository root):
    python -m database.scripts.load_sql             # schema + seed + load all rows
    python -m database.scripts.load_sql --limit 500 # quick smoke test

Steps
  1. run sql/01_schema.sql and sql/02_seed_reference_data.sql
  2. load data/processed/hourly_observed.csv  -> hourly_readings
  3. unpivot the four energy columns            -> circuit_energy (valid hours only)
  4. register the Task-1 champion model        -> ml_models
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import pymysql

from database import config as C
from database.scripts.sqlutil import run_script

CIRCUIT_COLS = {1: "Sub_metering_1", 2: "Sub_metering_2", 3: "Sub_metering_3", 4: "Other_energy_Wh"}
BATCH = 5000


def chunks(rows, n=BATCH):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def none_if_nan(v):
    return None if pd.isna(v) else float(v)


def connect(database=None):
    return pymysql.connect(host=C.MYSQL_HOST, port=C.MYSQL_PORT, user=C.MYSQL_USER, password=C.MYSQL_PASSWORD,
                           database=database, charset="utf8mb4", autocommit=False)


def main(limit: int | None = None):
    t0 = time.time()
    df = pd.read_csv(C.HOURLY_CSV, parse_dates=["timestamp"])
    if limit:
        df = df.head(limit)
    print(f"read {len(df):,} rows from {C.HOURLY_CSV.name} ({df.timestamp.min()} -> {df.timestamp.max()})")

    conn = connect()
    for f in ("01_schema.sql", "02_seed_reference_data.sql"):
        print(f"ran {f}: {run_script(conn, C.SQL_DIR / f, C.MYSQL_DATABASE)} statements")
    conn.select_db(C.MYSQL_DATABASE)

    # ---- hourly_readings: explicit reading_id = row order (stable, reproducible ids)
    readings = [
        (i + 1, 1, r.timestamp.to_pydatetime(), none_if_nan(r.Global_active_power), none_if_nan(r.Global_reactive_power),
         none_if_nan(r.Voltage), none_if_nan(r.Global_intensity), int(r.n_obs), int(bool(r.is_valid_hour)))
        for i, r in enumerate(df.itertuples(index=False))
    ]
    sql = ("INSERT INTO hourly_readings (reading_id, household_id, reading_ts, global_active_power_kw, "
           "global_reactive_power_kvar, voltage_v, global_intensity_a, n_obs_minutes, is_valid_hour) "
           "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)")
    with conn.cursor() as cur:
        for part in chunks(readings):
            cur.executemany(sql, part)
    conn.commit()
    print(f"hourly_readings: {len(readings):,} rows")

    # ---- circuit_energy (long format; hours with no measurements have no rows)
    energy = []
    for i, r in enumerate(df.itertuples(index=False)):
        for cid, col in CIRCUIT_COLS.items():
            v = getattr(r, col)
            if not pd.isna(v):
                energy.append((i + 1, cid, float(v)))
    with conn.cursor() as cur:
        for part in chunks(energy):
            cur.executemany("INSERT INTO circuit_energy (reading_id, circuit_id, energy_wh) VALUES (%s,%s,%s)", part)
    conn.commit()
    print(f"circuit_energy: {len(energy):,} rows")

    # ---- ml_models: the Task-1 champion (E4, tuned Random Forest) as reported in the notebook
    params = {"n_estimators": 50, "min_samples_leaf": 15, "max_features": 0.5, "max_depth": 12}
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ml_models (experiment_id, model_name, version, target, horizon_hours, hyperparameters, "
            "val_mae, test_mae, test_rmse, test_r2) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("E4", "Random Forest (tuned)", "1.0", "Global_active_power", 1, json.dumps(params),
             0.3502, 0.3046, 0.4474, 0.6004))
    conn.commit()
    conn.close()
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
