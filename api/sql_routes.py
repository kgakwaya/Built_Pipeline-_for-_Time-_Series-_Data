"""MySQL endpoints (prefix ``/sql``).

A reading is stored across two tables (see database/sql/01_schema.sql):
  * ``hourly_readings`` - one row per hour with the grid-level measurements
  * ``circuit_energy``  - one row per circuit per hour (sub-meters + derived "other")
Every write touches both tables inside one transaction; every read pivots them back
into the flat ``Reading`` record with conditional aggregation.
"""
from __future__ import annotations

from datetime import datetime

import pymysql
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.db import get_sql_conn
from api.schemas import (MEASUREMENT_FIELDS, Forecast, ForecastCreate, Message, Reading, ReadingCreate,
                         ReadingUpdate, finalize)
from database import config as C

router = APIRouter(prefix="/sql", tags=["MySQL"])
HOUSEHOLD = C.DEFAULT_HOUSEHOLD_ID
CIRCUIT_CODES = {"sub_metering_1_wh": "sub_metering_1", "sub_metering_2_wh": "sub_metering_2",
                 "sub_metering_3_wh": "sub_metering_3", "other_energy_wh": "other"}

_PIVOT = ",\n       ".join(f"MAX(CASE WHEN c.code = '{code}' THEN e.energy_wh END) AS {field}"
                           for field, code in CIRCUIT_CODES.items())
SELECT_READINGS = f"""
SELECT r.reading_id AS id, r.household_id, r.reading_ts AS timestamp,
       r.global_active_power_kw, r.global_reactive_power_kvar, r.voltage_v, r.global_intensity_a,
       {_PIVOT},
       r.n_obs_minutes, r.is_valid_hour
FROM hourly_readings r
LEFT JOIN circuit_energy e ON e.reading_id = r.reading_id
LEFT JOIN circuits c       ON c.circuit_id = e.circuit_id
WHERE {{where}}
GROUP BY r.reading_id
ORDER BY r.reading_ts {{order}}
"""

SELECT_FORECASTS = """
SELECT f.forecast_id AS id, f.household_id, f.made_at, f.target_ts AS target_timestamp,
       f.predicted_kw, f.actual_kw, m.experiment_id, m.version AS model_version, m.model_name
FROM forecasts f JOIN ml_models m ON m.model_id = f.model_id
WHERE {where}
ORDER BY f.target_ts DESC
"""


# ----------------------------------------------------------------------------- helpers
def _query(conn, sql: str, params=()) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def _fetch_one(conn, reading_id: int) -> dict | None:
    rows = _query(conn, SELECT_READINGS.format(where="r.reading_id = %s", order="ASC"), (reading_id,))
    return rows[0] if rows else None


def _write_energy(cur, reading_id: int, record: dict) -> None:
    """Replace the circuit_energy rows of a reading (NULL energies are simply not stored)."""
    cur.execute("SELECT code, circuit_id FROM circuits WHERE household_id = %s", (HOUSEHOLD,))
    circuit_id = {row["code"]: row["circuit_id"] for row in cur.fetchall()}
    cur.execute("DELETE FROM circuit_energy WHERE reading_id = %s", (reading_id,))
    rows = [(reading_id, circuit_id[code], record[field])
            for field, code in CIRCUIT_CODES.items() if record.get(field) is not None]
    if rows:
        cur.executemany("INSERT INTO circuit_energy (reading_id, circuit_id, energy_wh) VALUES (%s, %s, %s)", rows)


def _finalize(record: dict, changed: set[str]) -> dict:
    try:
        return finalize(record, changed)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _db_error(conn, exc: pymysql.err.MySQLError) -> HTTPException:
    conn.rollback()
    code = exc.args[0] if exc.args else None
    if code == 1062:   # ER_DUP_ENTRY on uq_readings_household_ts
        return HTTPException(status.HTTP_409_CONFLICT, "a reading for this household and hour already exists")
    if code == 3819:   # CHECK constraint violated
        return HTTPException(422, f"value rejected by a CHECK constraint: {exc.args[1]}")
    raise exc


def _check_range(start: datetime, end: datetime) -> None:
    if start > end:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "start must be before or equal to end")


# ----------------------------------------------------------------------------- readings: CRUD
@router.post("/readings", response_model=Reading, status_code=status.HTTP_201_CREATED,
             responses={409: {"model": Message}, 422: {"model": Message}})
def create_reading(body: ReadingCreate, conn=Depends(get_sql_conn)):
    """Insert one hourly reading (409 if that hour already exists)."""
    record = _finalize(body.model_dump(), body.model_fields_set)
    cols = ["household_id", "reading_ts", *MEASUREMENT_FIELDS, "n_obs_minutes", "is_valid_hour"]
    values = [HOUSEHOLD, record["timestamp"], *(record[f] for f in MEASUREMENT_FIELDS),
              record["n_obs_minutes"], record["is_valid_hour"]]
    try:
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO hourly_readings ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})", values)
            reading_id = cur.lastrowid
            _write_energy(cur, reading_id, record)
        conn.commit()
    except (pymysql.err.IntegrityError, pymysql.err.OperationalError) as exc:
        raise _db_error(conn, exc) from exc
    return _fetch_one(conn, reading_id)


@router.get("/readings", response_model=list[Reading])
def list_readings(limit: int = Query(100, ge=1, le=10_000), offset: int = Query(0, ge=0),
                  conn=Depends(get_sql_conn)):
    """Readings in chronological order, paginated with limit/offset."""
    sql = SELECT_READINGS.format(where="r.household_id = %s", order="ASC") + " LIMIT %s OFFSET %s"
    return _query(conn, sql, (HOUSEHOLD, limit, offset))


@router.get("/readings/latest", response_model=Reading, responses={404: {"model": Message}})
def latest_reading(valid_only: bool = Query(False, description="skip hours that failed the 45-minute quality rule"),
                   conn=Depends(get_sql_conn)):
    """Most recent reading. With valid_only=true, the most recent hour that has usable measurements."""
    inner = "SELECT reading_id FROM hourly_readings WHERE household_id = %s"
    if valid_only:
        inner += " AND is_valid_hour = 1"
    inner += " ORDER BY reading_ts DESC LIMIT 1"
    # resolve the id first (uses the (household_id, reading_ts) index) and pivot only that row
    rows = _query(conn, SELECT_READINGS.format(where=f"r.reading_id = ({inner})", order="ASC"), (HOUSEHOLD,))
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no readings found")
    return rows[0]


@router.get("/readings/range", response_model=list[Reading], responses={400: {"model": Message}})
def readings_by_range(start: datetime = Query(..., examples=["2010-11-20T00:00:00"]),
                      end: datetime = Query(..., examples=["2010-11-26T23:00:00"]),
                      valid_only: bool = False, limit: int = Query(10_000, ge=1, le=50_000),
                      conn=Depends(get_sql_conn)):
    """Readings with start <= timestamp <= end (both inclusive), oldest first."""
    _check_range(start, end)
    where = "r.household_id = %s AND r.reading_ts BETWEEN %s AND %s" + (" AND r.is_valid_hour = 1" if valid_only else "")
    return _query(conn, SELECT_READINGS.format(where=where, order="ASC") + " LIMIT %s", (HOUSEHOLD, start, end, limit))


@router.get("/readings/{reading_id}", response_model=Reading, responses={404: {"model": Message}})
def get_reading(reading_id: int, conn=Depends(get_sql_conn)):
    row = _fetch_one(conn, reading_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"reading {reading_id} not found")
    return row


@router.put("/readings/{reading_id}", response_model=Reading,
            responses={404: {"model": Message}, 409: {"model": Message}, 422: {"model": Message}})
def update_reading(reading_id: int, body: ReadingUpdate, conn=Depends(get_sql_conn)):
    """Update the fields that are sent; is_valid_hour and other_energy_wh are recomputed."""
    current = _fetch_one(conn, reading_id)
    if current is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"reading {reading_id} not found")
    patch = body.model_dump(exclude_unset=True)
    record = _finalize({**current, **patch}, set(patch))
    try:
        with conn.cursor() as cur:
            assignments = ", ".join(f"{f} = %s" for f in MEASUREMENT_FIELDS)
            cur.execute(f"UPDATE hourly_readings SET reading_ts = %s, {assignments}, n_obs_minutes = %s, "
                        "is_valid_hour = %s WHERE reading_id = %s",
                        (record["timestamp"], *(record[f] for f in MEASUREMENT_FIELDS),
                         record["n_obs_minutes"], record["is_valid_hour"], reading_id))
            _write_energy(cur, reading_id, record)
        conn.commit()
    except (pymysql.err.IntegrityError, pymysql.err.OperationalError) as exc:
        raise _db_error(conn, exc) from exc
    return _fetch_one(conn, reading_id)


@router.delete("/readings/{reading_id}", response_model=Message, responses={404: {"model": Message}})
def delete_reading(reading_id: int, conn=Depends(get_sql_conn)):
    """Delete a reading; its circuit_energy rows go with it (ON DELETE CASCADE)."""
    with conn.cursor() as cur:
        deleted = cur.execute("DELETE FROM hourly_readings WHERE reading_id = %s", (reading_id,))
    conn.commit()
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"reading {reading_id} not found")
    return {"detail": f"reading {reading_id} deleted"}


# ----------------------------------------------------------------------------- forecasts (used by Task 4)
@router.post("/forecasts", response_model=Forecast, status_code=status.HTTP_201_CREATED,
             responses={404: {"model": Message}})
def save_forecast(body: ForecastCreate, conn=Depends(get_sql_conn)):
    """Store a prediction. Re-forecasting the same hour with the same model replaces the old value."""
    model = _query(conn, "SELECT model_id FROM ml_models WHERE experiment_id = %s AND version = %s",
                   (body.experiment_id, body.model_version))
    if not model:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"model {body.experiment_id} v{body.model_version} is not registered")
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO forecasts (household_id, model_id, made_at, target_ts, predicted_kw, actual_kw) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE made_at = VALUES(made_at), "
            "predicted_kw = VALUES(predicted_kw), actual_kw = COALESCE(VALUES(actual_kw), actual_kw)",
            (HOUSEHOLD, model[0]["model_id"], body.made_at, body.target_timestamp, body.predicted_kw, body.actual_kw))
    conn.commit()
    where = "f.household_id = %s AND f.model_id = %s AND f.target_ts = %s"
    return _query(conn, SELECT_FORECASTS.format(where=where), (HOUSEHOLD, model[0]["model_id"], body.target_timestamp))[0]


@router.get("/forecasts", response_model=list[Forecast])
def list_forecasts(limit: int = Query(20, ge=1, le=1000), conn=Depends(get_sql_conn)):
    """Most recent forecasts first."""
    return _query(conn, SELECT_FORECASTS.format(where="f.household_id = %s") + " LIMIT %s", (HOUSEHOLD, limit))
