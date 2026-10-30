"""Locate, load and validate the raw household power consumption data.

The raw file is semicolon separated, uses ``?`` for missing measurements, stores
dates as ``d/m/Y`` and times as ``H:M:S``. Nothing here ever writes to the raw file.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

RAW_FILENAME = "household_power_consumption.txt"
EXPECTED_COLUMNS = [
    "Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
NUMERIC_COLUMNS = EXPECTED_COLUMNS[2:]
MISSING_TOKENS = ["?"]          # documented missing-value marker of the source file
DATE_FORMAT = "%d/%m/%Y"
TIME_FORMAT = "%H:%M:%S"


def find_project_root(start: Path | None = None) -> Path:
    """Return the project root (the folder containing ``src/`` and ``requirements.txt``).

    Works when called from the repository root, from ``notebooks/`` or any sub-folder,
    so no machine-specific absolute path is needed.
    """
    start = Path(start) if start is not None else Path.cwd()
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "src" / "data_loading.py").exists() and (candidate / "requirements.txt").exists():
            return candidate
    # Fallback: location of this file (src/..)
    return Path(__file__).resolve().parents[1]


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, used to prove that the raw data was never modified."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_archive(zip_path: Path) -> pd.DataFrame:
    """List the members of a zip archive (name, size) *before* extracting anything."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"Corrupt member in archive: {bad}")
        rows = [{"member": i.filename, "uncompressed_MB": round(i.file_size / 2**20, 2),
                 "compressed_MB": round(i.compress_size / 2**20, 2)} for i in archive.infolist()]
    return pd.DataFrame(rows)


def extract_dataset(zip_path: Path, destination: Path, member: str = RAW_FILENAME) -> Path:
    """Extract only the expected member into ``destination`` (never overwrites an existing file)."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / member
    if target.exists():
        return target
    with zipfile.ZipFile(zip_path) as archive:
        if member not in archive.namelist():
            raise FileNotFoundError(f"{member} not found in {zip_path}; members: {archive.namelist()}")
        with archive.open(member) as src, open(target, "wb") as dst:
            dst.write(src.read())
    return target


def locate_raw_file(raw_dir: Path) -> Path:
    """Return the raw dataset path or raise an actionable error."""
    path = Path(raw_dir) / RAW_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\nPlace the original '{RAW_FILENAME}' (extracted from the "
            f"supplied archive) in '{Path(raw_dir)}'. See data/README.md for details."
        )
    return path


def load_raw_data(path: Path) -> pd.DataFrame:
    """Read the raw file with the correct delimiter, dtypes and missing-value tokens.

    Date and Time stay as text (parsed separately); the seven measurements become float64.
    A ValueError is raised by pandas if an unexpected non-numeric token is present.
    """
    dtypes = {"Date": "string", "Time": "string", **{c: "float64" for c in NUMERIC_COLUMNS}}
    df = pd.read_csv(path, sep=";", na_values=MISSING_TOKENS, dtype=dtypes)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Assert that the columns match the documented schema exactly."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns {list(df.columns)}; expected {EXPECTED_COLUMNS}")


def iter_token_audit(path: Path, chunksize: int = 250_000) -> Iterator[pd.DataFrame]:
    """Chunked, memory-efficient audit of raw text tokens in the numeric columns."""
    for chunk in pd.read_csv(path, sep=";", dtype="string", chunksize=chunksize):
        yield chunk


def audit_numeric_tokens(path: Path, chunksize: int = 250_000) -> pd.DataFrame:
    """Count, per numeric column, tokens that are missing markers or non-numeric (excluding ``?``).

    Returns a table with columns: ``missing_token`` (``?``), ``empty`` and ``unexpected_nonnumeric``.
    """
    totals = {c: {"missing_token": 0, "empty": 0, "unexpected_nonnumeric": 0} for c in NUMERIC_COLUMNS}
    for chunk in iter_token_audit(path, chunksize):
        for col in NUMERIC_COLUMNS:
            s = chunk[col]
            is_q = s.isin(MISSING_TOKENS)
            is_empty = s.isna() | (s.str.strip() == "")
            parsed = pd.to_numeric(s, errors="coerce")
            bad = parsed.isna() & ~is_q & ~is_empty
            totals[col]["missing_token"] += int(is_q.sum())
            totals[col]["empty"] += int(is_empty.sum())
            totals[col]["unexpected_nonnumeric"] += int(bad.sum())
    return pd.DataFrame(totals).T


def parse_timestamps(date: pd.Series, time: pd.Series) -> pd.Series:
    """Combine Date (d/m/Y) and Time (H:M:S) into datetime64; unparseable values become NaT."""
    combined = date.astype("string").str.strip() + " " + time.astype("string").str.strip()
    return pd.to_datetime(combined, format=f"{DATE_FORMAT} {TIME_FORMAT}", errors="coerce")


def build_minute_frame(raw: pd.DataFrame, timestamps: pd.Series | None = None) -> pd.DataFrame:
    """Return a copy of ``raw`` indexed by ``timestamp`` (numeric columns only), sorted chronologically.

    Rows whose timestamp cannot be parsed are *not* silently dropped: a ValueError is raised.
    """
    validate_schema(raw)
    ts = timestamps if timestamps is not None else parse_timestamps(raw["Date"], raw["Time"])
    n_bad = int(ts.isna().sum())
    if n_bad:
        raise ValueError(f"{n_bad} rows have invalid timestamps; inspect them before continuing.")
    frame = raw[NUMERIC_COLUMNS].copy()
    frame.index = pd.DatetimeIndex(ts, name="timestamp")
    return frame.sort_index(kind="stable")


def data_quality_summary(frame: pd.DataFrame) -> dict:
    """Basic data-quality figures for an indexed frame (all computed, never typed)."""
    idx = frame.index
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "start": idx.min(),
        "end": idx.max(),
        "unique_timestamps": int(idx.nunique()),
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "duplicate_rows": int(frame.reset_index().duplicated().sum()),
        "rows_with_any_missing": int(frame.isna().any(axis=1).sum()),
        "total_missing_cells": int(frame.isna().sum().sum()),
    }
