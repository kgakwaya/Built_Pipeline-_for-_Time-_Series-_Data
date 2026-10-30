"""Database connections, provided to the routers as FastAPI dependencies.

Settings come from ``database/config.py`` (environment variables / ``.env``), the same
module used by the Task 2 loaders, so the API always talks to the databases they filled.
Tests replace these dependencies with in-memory fakes (``app.dependency_overrides``).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterator

import pymysql
from pymongo import MongoClient
from pymongo.database import Database

from database import config as C


def get_sql_conn() -> Iterator[pymysql.connections.Connection]:
    """One MySQL connection per request; rows come back as dicts. Closed after the response."""
    conn = pymysql.connect(host=C.MYSQL_HOST, port=C.MYSQL_PORT, user=C.MYSQL_USER, password=C.MYSQL_PASSWORD,
                           database=C.MYSQL_DATABASE, charset="utf8mb4", autocommit=False,
                           cursorclass=pymysql.cursors.DictCursor, connect_timeout=5)
    try:
        yield conn
    finally:
        conn.close()


@lru_cache(maxsize=1)
def _mongo_client() -> MongoClient:
    # MongoClient keeps its own connection pool, so one instance is shared by all requests.
    return MongoClient(C.MONGO_URI, serverSelectionTimeoutMS=5000)


def get_mongo_db() -> Database:
    return _mongo_client()[C.MONGO_DB]
