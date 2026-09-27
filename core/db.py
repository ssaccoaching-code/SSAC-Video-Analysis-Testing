"""Saving results to a database.

For now a SQLite file, results/splitsecond.db, which needs no server and
opens in any SQLite viewer. Two tables:

  runs     one row per analysis: when, which clip, swimmer, phase, pool
  metrics  one row per number: run_id, module, metric, value (text for
           lists and dicts, stored as JSON)

    from core.db import save_run, load_runs, load_metrics, export_csv
    run_id = save_run(results, clip_name="IMG_8090.MOV", swimmer="Theo", phase="lap", course="SCY")

Moving to a real database later means swapping the connect() call and the
CREATE TABLE text; nothing else in the app knows it is SQLite.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime
from typing import Any

from core.paths import RESULTS
from core.types import Result

DB_PATH = RESULTS / "splitsecond.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    clip TEXT NOT NULL,
    swimmer TEXT,
    phase TEXT NOT NULL,
    course TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    module TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    value_text TEXT
);
CREATE INDEX IF NOT EXISTS metrics_run ON metrics(run_id);
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    target = path or str(DB_PATH)
    RESULTS.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def save_run(results: list[Result], clip_name: str, swimmer: str = "", phase: str = "", course: str = "",
             notes: str = "", path: str | None = None) -> int:
    """Store one analysis. Returns the new run id."""
    conn = connect(path)
    try:
        cur = conn.execute(
            "INSERT INTO runs (run_at, clip, swimmer, phase, course, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), clip_name, swimmer or None, phase, course or None, notes or None),
        )
        run_id = int(cur.lastrowid)
        from core.report import all_results

        rows = []
        for result in all_results(results):
            for key, value in result.metrics.items():
                if value is None:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    rows.append((run_id, result.module, key, float(value), None))
                else:
                    rows.append((run_id, result.module, key, None, json.dumps(value)))
        conn.executemany("INSERT INTO metrics (run_id, module, metric, value, value_text) VALUES (?, ?, ?, ?, ?)", rows)
        conn.commit()
        return run_id
    finally:
        conn.close()


def load_runs(path: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    conn = connect(path)
    try:
        cur = conn.execute("SELECT id, run_at, clip, swimmer, phase, course, notes FROM runs ORDER BY id DESC LIMIT ?", (limit,))
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def load_metrics(run_id: int | None = None, path: str | None = None) -> list[dict[str, Any]]:
    """Long-form rows: one per metric, joined with the run details."""
    conn = connect(path)
    try:
        sql = ("SELECT r.id AS run_id, r.run_at, r.clip, r.swimmer, r.phase, r.course, m.module, m.metric, m.value, m.value_text "
               "FROM metrics m JOIN runs r ON r.id = m.run_id")
        args: tuple = ()
        if run_id is not None:
            sql += " WHERE r.id = ?"
            args = (run_id,)
        sql += " ORDER BY r.id DESC, m.id"
        cur = conn.execute(sql, args)
        names = [d[0] for d in cur.description]
        out = []
        for row in cur.fetchall():
            item = dict(zip(names, row))
            if item["value"] is None and item["value_text"]:
                item["value"] = json.loads(item["value_text"])
            item.pop("value_text", None)
            out.append(item)
        return out
    finally:
        conn.close()


def delete_run(run_id: int, path: str | None = None) -> None:
    conn = connect(path)
    try:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.commit()
    finally:
        conn.close()


def export_csv(path: str | None = None) -> str:
    """Every saved metric as CSV text (one row per metric)."""
    rows = load_metrics(path=path)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["run_id", "run_at", "clip", "swimmer", "phase", "course", "module", "metric", "value"])
    for r in rows:
        value = r["value"]
        writer.writerow([r["run_id"], r["run_at"], r["clip"], r["swimmer"] or "", r["phase"], r["course"] or "",
                         r["module"], r["metric"], json.dumps(value) if isinstance(value, (list, dict)) else value])
    return buf.getvalue()
