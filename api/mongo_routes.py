"""MongoDB endpoints (prefix ``/mongo``).

Documents keep the Task 2 design (database/scripts/load_mongo.py): one document per hour with
embedded ``measurements``, ``sub_metering_wh``, ``quality`` and ``calendar`` sub-documents.
The API converts them to / from the same flat ``Reading`` record used by the SQL endpoints.
"""
from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError, WriteError

from api.db import get_mongo_db
from api.schemas import (MEASUREMENT_FIELDS, Forecast, ForecastCreate, Message, Reading, ReadingCreate,
                         ReadingUpdate, finalize)
from database import config as C

router = APIRouter(prefix="/mongo", tags=["MongoDB"])
HOUSEHOLD = C.DEFAULT_HOUSEHOLD_CODE
# flat API field -> key inside the "sub_metering_wh" sub-document
SUB_KEYS = {"sub_metering_1_wh": "kitchen", "sub_metering_2_wh": "laundry_room",
            "sub_metering_3_wh": "water_heater_ac", "other_energy_wh": "other"}


# ----------------------------------------------------------------------------- document <-> record
def to_document(record: dict) -> dict:
    ts: datetime = record["timestamp"]
    return {
        "household_id": HOUSEHOLD,
        "timestamp": ts,
        "measurements": {f: record.get(f) for f in MEASUREMENT_FIELDS},
        "sub_metering_wh": {key: record.get(f) for f, key in SUB_KEYS.items()},
        "quality": {"n_obs_minutes": int(record["n_obs_minutes"]), "is_valid_hour": bool(record["is_valid_hour"])},
        "calendar": {"year": ts.year, "month": ts.month, "day": ts.day, "hour": ts.hour,
                     "weekday": ts.weekday(), "is_weekend": ts.weekday() >= 5},
    }


def to_record(doc: dict) -> dict:
    m, s, q = doc.get("measurements", {}), doc.get("sub_metering_wh", {}), doc.get("quality", {})
    return {"id": str(doc["_id"]), "household_id": doc["household_id"], "timestamp": doc["timestamp"],
            **{f: m.get(f) for f in MEASUREMENT_FIELDS},
            **{f: s.get(key) for f, key in SUB_KEYS.items()},
            "n_obs_minutes": q.get("n_obs_minutes", 0), "is_valid_hour": q.get("is_valid_hour", False)}


def _oid(reading_id: str) -> ObjectId:
    try:
        return ObjectId(reading_id)
    except (InvalidId, TypeError) as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"reading {reading_id} not found") from exc


def _finalize(record: dict, changed: set[str]) -> dict:
    try:
        return finalize(record, changed)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _write_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DuplicateKeyError):   # unique index uq_household_ts
        return HTTPException(status.HTTP_409_CONFLICT, "a reading for this household and hour already exists")
    if isinstance(exc, WriteError) and exc.code == 121:   # $jsonSchema validator rejected the document
        return HTTPException(422, "document failed collection schema validation")
    raise exc


# ----------------------------------------------------------------------------- readings: CRUD
@router.post("/readings", response_model=Reading, status_code=status.HTTP_201_CREATED,
             responses={409: {"model": Message}, 422: {"model": Message}})
def create_reading(body: ReadingCreate, db=Depends(get_mongo_db)):
    """Insert one hourly document (409 if that hour already exists)."""
    doc = to_document(_finalize(body.model_dump(), body.model_fields_set))
    try:
        doc["_id"] = db.hourly_readings.insert_one(doc).inserted_id
    except (DuplicateKeyError, WriteError) as exc:
        raise _write_error(exc) from exc
    return to_record(doc)


@router.get("/readings", response_model=list[Reading])
def list_readings(limit: int = Query(100, ge=1, le=10_000), offset: int = Query(0, ge=0), db=Depends(get_mongo_db)):
    """Readings in chronological order, paginated with limit/offset."""
    cursor = db.hourly_readings.find({"household_id": HOUSEHOLD}).sort("timestamp", ASCENDING).skip(offset).limit(limit)
    return [to_record(d) for d in cursor]


@router.get("/readings/latest", response_model=Reading, responses={404: {"model": Message}})
def latest_reading(valid_only: bool = Query(False, description="skip hours that failed the 45-minute quality rule"),
                   db=Depends(get_mongo_db)):
    """Most recent document. With valid_only=true, the most recent hour that has usable measurements."""
    query = {"household_id": HOUSEHOLD, **({"quality.is_valid_hour": True} if valid_only else {})}
    doc = db.hourly_readings.find_one(query, sort=[("timestamp", DESCENDING)])
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no readings found")
    return to_record(doc)


@router.get("/readings/range", response_model=list[Reading], responses={400: {"model": Message}})
def readings_by_range(start: datetime = Query(..., examples=["2010-11-20T00:00:00"]),
                      end: datetime = Query(..., examples=["2010-11-26T23:00:00"]),
                      valid_only: bool = False, limit: int = Query(10_000, ge=1, le=50_000),
                      db=Depends(get_mongo_db)):
    """Documents with start <= timestamp <= end (both inclusive), oldest first."""
    if start > end:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "start must be before or equal to end")
    query = {"household_id": HOUSEHOLD, "timestamp": {"$gte": start, "$lte": end}}
    if valid_only:
        query["quality.is_valid_hour"] = True
    return [to_record(d) for d in db.hourly_readings.find(query).sort("timestamp", ASCENDING).limit(limit)]


@router.get("/readings/{reading_id}", response_model=Reading, responses={404: {"model": Message}})
def get_reading(reading_id: str, db=Depends(get_mongo_db)):
    doc = db.hourly_readings.find_one({"_id": _oid(reading_id)})
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"reading {reading_id} not found")
    return to_record(doc)


@router.put("/readings/{reading_id}", response_model=Reading,
            responses={404: {"model": Message}, 409: {"model": Message}, 422: {"model": Message}})
def update_reading(reading_id: str, body: ReadingUpdate, db=Depends(get_mongo_db)):
    """Update the fields that are sent; is_valid_hour, other_energy_wh and calendar are recomputed."""
    oid = _oid(reading_id)
    current = db.hourly_readings.find_one({"_id": oid})
    if current is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"reading {reading_id} not found")
    patch = body.model_dump(exclude_unset=True)
    doc = to_document(_finalize({**to_record(current), **patch}, set(patch)))
    try:
        updated = db.hourly_readings.find_one_and_update({"_id": oid}, {"$set": doc},
                                                         return_document=ReturnDocument.AFTER)
    except (DuplicateKeyError, WriteError) as exc:
        raise _write_error(exc) from exc
    return to_record(updated)


@router.delete("/readings/{reading_id}", response_model=Message, responses={404: {"model": Message}})
def delete_reading(reading_id: str, db=Depends(get_mongo_db)):
    if db.hourly_readings.delete_one({"_id": _oid(reading_id)}).deleted_count == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"reading {reading_id} not found")
    return {"detail": f"reading {reading_id} deleted"}


# ----------------------------------------------------------------------------- forecasts (used by Task 4)
def _forecast_record(doc: dict) -> dict:
    return {"id": str(doc["_id"]), "household_id": doc["household_id"], "made_at": doc["made_at"],
            "target_timestamp": doc["target_timestamp"], "predicted_kw": doc["predicted_kw"],
            "actual_kw": doc.get("actual_kw"), "experiment_id": doc["model"]["experiment_id"],
            "model_version": doc["model"]["version"], "model_name": doc["model"].get("name")}


@router.post("/forecasts", response_model=Forecast, status_code=status.HTTP_201_CREATED,
             responses={404: {"model": Message}})
def save_forecast(body: ForecastCreate, db=Depends(get_mongo_db)):
    """Store a prediction. Re-forecasting the same hour with the same model replaces the old value."""
    model = db.models.find_one({"experiment_id": body.experiment_id, "version": body.model_version})
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"model {body.experiment_id} v{body.model_version} is not registered")
    key = {"household_id": HOUSEHOLD, "model.experiment_id": body.experiment_id,
           "model.version": body.model_version, "target_timestamp": body.target_timestamp}
    fields = {"made_at": body.made_at, "predicted_kw": body.predicted_kw,
              "horizon_hours": int((body.target_timestamp - body.made_at).total_seconds() // 3600),
              "created_at": datetime.now(timezone.utc).replace(tzinfo=None)}
    if body.actual_kw is not None:
        fields["actual_kw"] = body.actual_kw
    # on insert, the equality fields of `key` (household, model.*, target) seed the new document
    doc = db.forecasts.find_one_and_update(
        key, {"$set": fields, "$setOnInsert": {"model.name": model.get("model_name")}},
        upsert=True, return_document=ReturnDocument.AFTER)
    return _forecast_record(doc)


@router.get("/forecasts", response_model=list[Forecast])
def list_forecasts(limit: int = Query(20, ge=1, le=1000), db=Depends(get_mongo_db)):
    """Most recent forecasts first."""
    cursor = db.forecasts.find({"household_id": HOUSEHOLD}).sort("target_timestamp", DESCENDING).limit(limit)
    return [_forecast_record(d) for d in cursor]
