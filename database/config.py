"""Connection settings shared by the loaders, the query runners and the API.

Values come from environment variables (or a ``.env`` file in the repository
root) so no credentials are committed.  Defaults match docker-compose.yml.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[1]
HOURLY_CSV = Path(os.getenv("HOURLY_CSV", ROOT / "data" / "processed" / "hourly_observed.csv"))
SQL_DIR = ROOT / "database" / "sql"
RESULTS_DIR = ROOT / "database" / "results"

# ------------------------------------------------------------------ MySQL
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "power_db")


def mysql_url(database: str | None = MYSQL_DATABASE) -> str:
    """SQLAlchemy URL (PyMySQL driver)."""
    auth = f"{quote_plus(MYSQL_USER)}:{quote_plus(MYSQL_PASSWORD)}"
    db = f"/{database}" if database else ""
    return f"mysql+pymysql://{auth}@{MYSQL_HOST}:{MYSQL_PORT}{db}?charset=utf8mb4"


# ---------------------------------------------------------------- MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = os.getenv("MONGO_DB", "power_db")

DEFAULT_HOUSEHOLD_ID = 1            # SQL id
DEFAULT_HOUSEHOLD_CODE = "hh_001"   # Mongo id
