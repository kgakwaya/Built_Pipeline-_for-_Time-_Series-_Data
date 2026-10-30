"""Tiny helper: run a .sql script statement by statement through PyMySQL."""
from __future__ import annotations

import re
from pathlib import Path


def split_statements(sql: str) -> list[str]:
    """Split a script on ';' after removing '--' comments (no ';' inside string literals is assumed)."""
    no_comments = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in no_comments.split(";") if s.strip()]


def run_script(conn, path: Path, db_name: str | None = None) -> int:
    """Execute a script; if ``db_name`` is given the default database name ``power_db`` is replaced by it."""
    text = Path(path).read_text(encoding="utf-8")
    if db_name and db_name != "power_db":
        text = text.replace("power_db", db_name)
    stmts = split_statements(text)
    with conn.cursor() as cur:
        for s in stmts:
            cur.execute(s)
    conn.commit()
    return len(stmts)
