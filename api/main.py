"""FastAPI application for Task 3.

Run from the repository root:
    uvicorn api.main:app --reload            # interactive docs at http://127.0.0.1:8000/docs

Endpoints (identical for /sql and /mongo):
    POST   /{db}/readings                     create an hourly reading
    GET    /{db}/readings?limit=&offset=      list readings (chronological)
    GET    /{db}/readings/latest?valid_only=  latest record
    GET    /{db}/readings/range?start=&end=   records by date range (inclusive)
    GET    /{db}/readings/{id}                one reading
    PUT    /{db}/readings/{id}                update the supplied fields
    DELETE /{db}/readings/{id}                delete a reading
    POST   /{db}/forecasts, GET /{db}/forecasts   store / list predictions (Task 4)
"""
from __future__ import annotations

import pymysql
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pymongo.errors import ConnectionFailure

from api import mongo_routes, sql_routes
from api.db import get_mongo_db, get_sql_conn

app = FastAPI(
    title="Household Power Time-Series API",
    description="CRUD and time-series queries over the hourly household electricity readings, "
                "backed by both MySQL and MongoDB (Formative 1, Task 3).",
    version="1.0.0",
)
app.include_router(sql_routes.router)
app.include_router(mongo_routes.router)


@app.exception_handler(pymysql.err.OperationalError)
async def mysql_unavailable(request: Request, exc: pymysql.err.OperationalError):
    """Connection / login failures (the routes handle constraint errors themselves) -> 503."""
    return JSONResponse(status_code=503, content={"detail": f"MySQL unavailable: {exc.args[-1]}"})


@app.exception_handler(ConnectionFailure)
async def mongo_unavailable(request: Request, exc: ConnectionFailure):
    return JSONResponse(status_code=503, content={"detail": "MongoDB unavailable: cannot reach " + exc.__class__.__name__})


@app.get("/", tags=["health"])
def root():
    return {"service": app.title, "docs": "/docs", "backends": ["/sql", "/mongo"]}


@app.get("/health", tags=["health"])
def health():
    """Reports whether each database is reachable (the API itself starts even if one is down)."""
    status = {}
    try:
        for conn in get_sql_conn():
            conn.ping()
        status["mysql"] = "ok"
    except Exception as exc:  # noqa: BLE001 - report any connection problem
        status["mysql"] = f"unavailable: {exc}"
    try:
        get_mongo_db().command("ping")
        status["mongodb"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["mongodb"] = f"unavailable: {exc}"
    return status
