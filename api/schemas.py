"""Request / response models shared by the SQL and MongoDB routers.

Both databases expose the *same* flat JSON record, so a client (e.g. the Task 4
prediction script) can switch backend without changing its parsing code:

    {"id": ..., "household_id": ..., "timestamp": "2010-11-26T20:00:00",
     "global_active_power_kw": 1.27, ..., "sub_metering_1_wh": 0.0, ..., "other_energy_wh": 850.1,
     "n_obs_minutes": 60, "is_valid_hour": true}

The derived fields follow the Task 1 preprocessing rules (``src/preprocessing.py``):
  * ``is_valid_hour``   = n_obs_minutes >= 45
  * ``other_energy_wh`` = global_active_power_kw * 1000 - (sub_metering_1 + 2 + 3)
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

MIN_OBS_MINUTES = 45   # same quality rule as Task 1 (aggregate_to_hourly(min_obs=45))
SUB_FIELDS = ("sub_metering_1_wh", "sub_metering_2_wh", "sub_metering_3_wh")
MEASUREMENT_FIELDS = ("global_active_power_kw", "global_reactive_power_kvar", "voltage_v", "global_intensity_a")
ENERGY_FIELDS = (*SUB_FIELDS, "other_energy_wh")
VALUE_FIELDS = (*MEASUREMENT_FIELDS, *ENERGY_FIELDS)


def _on_the_hour(ts: datetime | None) -> datetime | None:
    """Readings are hourly: the timestamp is the start of the hour and must carry no time zone."""
    if ts is None:
        return ts
    if ts.minute or ts.second or ts.microsecond:
        raise ValueError("timestamp must be at the start of an hour (e.g. 2010-11-26T20:00:00)")
    if ts.tzinfo is not None:
        raise ValueError("timestamp must be local clock time without a time zone (the dataset has none)")
    return ts


class ReadingValues(BaseModel):
    global_active_power_kw: float | None = Field(None, ge=0, description="TARGET: hourly mean active power (kW)")
    global_reactive_power_kvar: float | None = Field(None, ge=0)
    voltage_v: float | None = Field(None, ge=0)
    global_intensity_a: float | None = Field(None, ge=0)
    sub_metering_1_wh: float | None = Field(None, ge=0, description="kitchen energy during the hour (Wh)")
    sub_metering_2_wh: float | None = Field(None, ge=0, description="laundry room energy (Wh)")
    sub_metering_3_wh: float | None = Field(None, ge=0, description="water heater & A/C energy (Wh)")
    other_energy_wh: float | None = Field(None, ge=0, description="derived automatically when omitted")


class ReadingCreate(ReadingValues):
    timestamp: datetime
    n_obs_minutes: int = Field(60, ge=0, le=60, description="minutes observed in the hour")

    _check_ts = field_validator("timestamp")(_on_the_hour)

    model_config = {"json_schema_extra": {"example": {
        "timestamp": "2010-11-27T00:00:00", "global_active_power_kw": 1.25, "global_reactive_power_kvar": 0.11,
        "voltage_v": 240.3, "global_intensity_a": 5.2, "sub_metering_1_wh": 0.0, "sub_metering_2_wh": 30.0,
        "sub_metering_3_wh": 600.0, "n_obs_minutes": 60}}}


class ReadingUpdate(ReadingValues):
    """PUT body: only the fields that are sent are changed; derived fields are recomputed."""
    timestamp: datetime | None = None
    n_obs_minutes: int | None = Field(None, ge=0, le=60)

    _check_ts = field_validator("timestamp")(_on_the_hour)

    model_config = {"json_schema_extra": {"example": {"global_active_power_kw": 1.4, "sub_metering_3_wh": 650.0}}}


class Reading(ReadingValues):
    id: int | str
    household_id: int | str
    timestamp: datetime
    n_obs_minutes: int
    is_valid_hour: bool


class ForecastCreate(BaseModel):
    made_at: datetime = Field(description="forecast origin: last observed hour used as input")
    target_timestamp: datetime = Field(description="hour being predicted")
    predicted_kw: float = Field(ge=0)
    actual_kw: float | None = Field(None, ge=0)
    experiment_id: str = "E4"
    model_version: str = "1.0"


class Forecast(ForecastCreate):
    id: int | str
    household_id: int | str
    model_name: str | None = None


class Message(BaseModel):
    detail: str


def finalize(record: dict, changed: set[str]) -> dict:
    """Apply the Task 1 derivation rules to a complete record (in place) and return it.

    ``changed`` lists the fields the client sent (``model.model_fields_set``). ``other_energy_wh`` is
    recomputed when the client did not set it explicitly and the target or a sub-meter changed.
    """
    record["is_valid_hour"] = record.get("n_obs_minutes", 0) >= MIN_OBS_MINUTES
    inputs = ("global_active_power_kw", *SUB_FIELDS)
    if "other_energy_wh" not in changed and changed & set(inputs):
        if all(record.get(f) is not None for f in inputs):
            other = record["global_active_power_kw"] * 1000.0 - sum(record[f] for f in SUB_FIELDS)
            if other < 0:
                raise ValueError("sub-metering energy exceeds total active energy "
                                 "(global_active_power_kw * 1000 < sum of sub_metering_*_wh)")
            record["other_energy_wh"] = other
        else:
            record["other_energy_wh"] = None
    return record
