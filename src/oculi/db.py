from __future__ import annotations

import json
import sqlite3
import threading
from datetime import timedelta
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .template import FaceTemplate

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS watchlists (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subjects (
    id TEXT PRIMARY KEY,
    watchlist_id TEXT NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    engine TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    sha256 TEXT NOT NULL,
    det_score REAL NOT NULL DEFAULT 0,
    bbox TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_subjects_watchlist ON subjects(watchlist_id);
CREATE INDEX IF NOT EXISTS idx_templates_subject ON templates(subject_id);
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'rtsp',
    enabled INTEGER NOT NULL DEFAULT 1,
    watchlist_id TEXT NOT NULL DEFAULT 'default',
    milestone INTEGER NOT NULL DEFAULT 0,
    options TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    source_id TEXT,
    ts TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        with self.conn:
            self.conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # -- watchlists ---------------------------------------------------------
    def create_watchlist(self, watchlist_id: str, name: str) -> dict:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO watchlists(id, name, created_at) VALUES (?, ?, ?)",
                (watchlist_id, name, _now()),
            )
        row = self.get_watchlist(watchlist_id)
        assert row is not None
        return row

    def get_watchlist(self, watchlist_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT id, name, created_at FROM watchlists WHERE id = ?", (watchlist_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_watchlists(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT w.id, w.name, w.created_at, COUNT(t.id) AS template_count,
                       COUNT(DISTINCT s.id) AS subject_count
                FROM watchlists w
                LEFT JOIN subjects s ON s.watchlist_id = w.id
                LEFT JOIN templates t ON t.subject_id = s.id
                GROUP BY w.id ORDER BY w.created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    # -- subjects -----------------------------------------------------------
    def create_or_get_subject(
        self, subject_id: str, watchlist_id: str, name: str, metadata: dict | None = None
    ) -> dict:
        with self._lock, self.conn:
            row = self.conn.execute(
                "SELECT id, watchlist_id, name, metadata, created_at FROM subjects WHERE id = ?",
                (subject_id,),
            ).fetchone()
            if row:
                if row["watchlist_id"] != watchlist_id:
                    raise ValueError(
                        f"subject {subject_id!r} already belongs to watchlist {row['watchlist_id']!r}"
                    )
                self.conn.execute("UPDATE subjects SET name = ? WHERE id = ?", (name, subject_id))
            else:
                self.conn.execute(
                    "INSERT INTO subjects(id, watchlist_id, name, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (subject_id, watchlist_id, name, json.dumps(metadata or {}), _now()),
                )
        result = self.get_subject(subject_id)
        assert result is not None
        return result

    def get_subject(self, subject_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT id, watchlist_id, name, metadata, created_at FROM subjects WHERE id = ?",
                (subject_id,),
            ).fetchone()
        return self._subject_row(row) if row else None

    def list_subjects(self, watchlist_id: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT s.id, s.watchlist_id, s.name, s.metadata, s.created_at,
                       COUNT(t.id) AS template_count
                FROM subjects s LEFT JOIN templates t ON t.subject_id = s.id
                WHERE s.watchlist_id = ?
                GROUP BY s.id ORDER BY s.created_at
                """,
                (watchlist_id,),
            ).fetchall()
        return [self._subject_row(row, template_count=row["template_count"]) for row in rows]

    @staticmethod
    def _subject_row(row: sqlite3.Row, template_count: int | None = None) -> dict:
        data = {
            "id": row["id"],
            "watchlist_id": row["watchlist_id"],
            "name": row["name"],
            "metadata": json.loads(row["metadata"] or "{}"),
            "created_at": row["created_at"],
        }
        if template_count is not None:
            data["template_count"] = template_count
        return data

    # -- templates ----------------------------------------------------------
    def add_template(self, subject_id: str, template: FaceTemplate) -> int:
        with self._lock, self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO templates(subject_id, engine, model, dim, vector, sha256,
                                      det_score, bbox, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subject_id,
                    template.engine,
                    template.model,
                    template.dim,
                    template.vector_blob(),
                    template.vector_sha256,
                    float(template.det_score),
                    json.dumps(template.bbox) if template.bbox else None,
                    template.created_at,
                ),
            )
            return int(cursor.lastrowid)

    def add_templates(self, subject_id: str, templates: list[FaceTemplate]) -> list[int]:
        ids: list[int] = []
        with self._lock, self.conn:
            for template in templates:
                cursor = self.conn.execute(
                    """
                    INSERT INTO templates(subject_id, engine, model, dim, vector, sha256,
                                          det_score, bbox, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        subject_id,
                        template.engine,
                        template.model,
                        template.dim,
                        template.vector_blob(),
                        template.vector_sha256,
                        float(template.det_score),
                        json.dumps(template.bbox) if template.bbox else None,
                        template.created_at,
                    ),
                )
                ids.append(int(cursor.lastrowid))
        return ids

    def get_template(self, template_id: int) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT t.*, s.name AS subject_name, s.watchlist_id
                FROM templates t JOIN subjects s ON s.id = t.subject_id
                WHERE t.id = ?
                """,
                (template_id,),
            ).fetchone()
        return self._template_row(row, include_vector=True) if row else None

    def list_templates(self, subject_id: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT t.*, s.name AS subject_name, s.watchlist_id
                FROM templates t JOIN subjects s ON s.id = t.subject_id
                WHERE t.subject_id = ? ORDER BY t.id
                """,
                (subject_id,),
            ).fetchall()
        return [self._template_row(row) for row in rows]

    def watchlist_template_spec(self, watchlist_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT t.engine, t.model, t.dim
                FROM templates t JOIN subjects s ON s.id = t.subject_id
                WHERE s.watchlist_id = ? LIMIT 1
                """,
                (watchlist_id,),
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _template_row(row: sqlite3.Row, include_vector: bool = False) -> dict:
        template = FaceTemplate(
            vector=FaceTemplate.vector_from_blob(row["vector"], row["dim"]),
            engine=row["engine"],
            model=row["model"],
            det_score=row["det_score"],
            bbox=json.loads(row["bbox"]) if row["bbox"] else None,
            created_at=row["created_at"],
        )
        data = {
            "template_id": row["id"],
            "subject_id": row["subject_id"],
            "subject_name": row["subject_name"] if "subject_name" in row.keys() else None,
            "watchlist_id": row["watchlist_id"] if "watchlist_id" in row.keys() else None,
            "created_at": row["created_at"],
        }
        if include_vector:
            data["template"] = template.to_dict()
        else:
            data["engine"] = template.engine
            data["model"] = template.model
            data["dim"] = template.dim
            data["template_sha256"] = template.vector_sha256
            data["bbox"] = template.bbox
            data["det_score"] = template.det_score
        return data

    def template_results(self, template_ids: list[int]) -> dict[int, dict]:
        if not template_ids:
            return {}
        placeholders = ",".join("?" for _ in template_ids)
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT t.id, t.subject_id, t.dim, t.det_score, t.bbox, s.name AS subject_name,
                       s.watchlist_id
                FROM templates t JOIN subjects s ON s.id = t.subject_id
                WHERE t.id IN ({placeholders})
                """,
                template_ids,
            ).fetchall()
        return {
            int(row["id"]): {
                "template_id": int(row["id"]),
                "subject_id": row["subject_id"],
                "subject_name": row["subject_name"],
                "watchlist_id": row["watchlist_id"],
                "dim": row["dim"],
                "det_score": row["det_score"],
                "bbox": json.loads(row["bbox"]) if row["bbox"] else None,
            }
            for row in rows
        }

    def iter_index_rows(self, watchlist_id: str | None = None):
        with self._lock:
            if watchlist_id is None:
                rows = self.conn.execute(
                    """
                    SELECT t.id, t.dim, t.vector, s.watchlist_id
                    FROM templates t JOIN subjects s ON s.id = t.subject_id
                    """
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT t.id, t.dim, t.vector, s.watchlist_id
                    FROM templates t JOIN subjects s ON s.id = t.subject_id
                    WHERE s.watchlist_id = ?
                    """,
                    (watchlist_id,),
                ).fetchall()
        for row in rows:
            yield int(row["id"]), int(row["dim"]), row["watchlist_id"], row["vector"]

    def delete_subject(self, subject_id: str) -> list[int]:
        with self._lock, self.conn:
            rows = self.conn.execute(
                "SELECT id FROM templates WHERE subject_id = ?", (subject_id,)
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            self.conn.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
        return ids

    def delete_template(self, template_id: int) -> bool:
        with self._lock, self.conn:
            cursor = self.conn.execute("DELETE FROM templates WHERE id = ?", (template_id,))
        return cursor.rowcount > 0

    def delete_watchlist(self, watchlist_id: str) -> list[int]:
        with self._lock, self.conn:
            rows = self.conn.execute(
                """
                SELECT t.id FROM templates t JOIN subjects s ON s.id = t.subject_id
                WHERE s.watchlist_id = ?
                """,
                (watchlist_id,),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            self.conn.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))
        return ids

    # -- sources ------------------------------------------------------------
    def create_source(
        self,
        source_id: str,
        name: str,
        url: str,
        kind: str = "rtsp",
        enabled: bool = True,
        watchlist_id: str = "default",
        milestone: bool = False,
        options: dict | None = None,
    ) -> dict:
        now = _now()
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO sources(id, name, url, kind, enabled, watchlist_id, milestone,
                                    options, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, url=excluded.url, kind=excluded.kind,
                    enabled=excluded.enabled, watchlist_id=excluded.watchlist_id,
                    milestone=excluded.milestone, options=excluded.options,
                    updated_at=excluded.updated_at
                """,
                (
                    source_id, name, url, kind, int(enabled), watchlist_id, int(milestone),
                    json.dumps(options or {}), now, now,
                ),
            )
        result = self.get_source(source_id)
        assert result is not None
        return result

    def get_source(self, source_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM sources WHERE id = ?", (source_id,)
            ).fetchone()
        return self._source_row(row) if row else None

    def list_sources(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM sources ORDER BY created_at").fetchall()
        return [self._source_row(row) for row in rows]

    @staticmethod
    def _source_row(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "name": row["name"],
            "url": row["url"],
            "kind": row["kind"],
            "enabled": bool(row["enabled"]),
            "watchlist_id": row["watchlist_id"],
            "milestone": bool(row["milestone"]),
            "options": json.loads(row["options"] or "{}"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def set_source_enabled(self, source_id: str, enabled: bool) -> dict:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE sources SET enabled = ?, updated_at = ? WHERE id = ?",
                (int(enabled), _now(), source_id),
            )
        source = self.get_source(source_id)
        if not source:
            raise KeyError(f"source {source_id!r} not found")
        return source

    def delete_source(self, source_id: str) -> bool:
        with self._lock, self.conn:
            cursor = self.conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        return cursor.rowcount > 0

    # -- events -------------------------------------------------------------
    def add_event(self, payload: dict) -> dict:
        event_type = str(payload.get("type", "unknown"))
        source_id = payload.get("source_id")
        ts = str(payload.get("ts") or _now())
        body = json.dumps(payload, sort_keys=True, default=str)
        with self._lock, self.conn:
            cursor = self.conn.execute(
                "INSERT INTO events(type, source_id, ts, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (event_type, source_id, ts, body, _now()),
            )
            event_id = int(cursor.lastrowid)
        stored = dict(payload)
        stored["id"] = event_id
        stored.setdefault("ts", ts)
        # Keep the persisted JSON consistent with the returned envelope.
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE events SET payload = ? WHERE id = ?",
                (json.dumps(stored, sort_keys=True, default=str), event_id),
            )
        return stored

    def update_event_payload(self, event_id: int, payload: dict) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE events SET payload = ? WHERE id = ?",
                (json.dumps(payload, sort_keys=True, default=str), int(event_id)),
            )

    def list_events(
        self,
        after_id: int = 0,
        limit: int = 100,
        event_type: str | None = None,
        source_id: str | None = None,
    ) -> list[dict]:
        clauses = ["id > ?"]
        params: list = [int(after_id)]
        if event_type:
            clauses.append("type = ?")
            params.append(event_type)
        if source_id:
            clauses.append("source_id = ?")
            params.append(source_id)
        params.append(max(1, min(int(limit), 1000)))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id, payload FROM events WHERE {' AND '.join(clauses)} ORDER BY id LIMIT ?",
                params,
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def event_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT type, COUNT(*) AS n FROM events GROUP BY type ORDER BY n DESC"
            ).fetchall()
        return {row["type"]: int(row["n"]) for row in rows}

    def prune_events(self, days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(0, int(days)))).isoformat()
        with self._lock, self.conn:
            cursor = self.conn.execute("DELETE FROM events WHERE created_at < ?", (cutoff,))
        return int(cursor.rowcount)

    def backup(self, destination: str | Path) -> str:
        target = str(destination)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            out = sqlite3.connect(target)
            try:
                self.conn.backup(out)
            finally:
                out.close()
        return target

    def counts(self) -> dict[str, int]:
        with self._lock:
            watchlists = self.conn.execute("SELECT COUNT(*) FROM watchlists").fetchone()[0]
            subjects = self.conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
            templates = self.conn.execute("SELECT COUNT(*) FROM templates").fetchone()[0]
            sources = self.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            events = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {
            "watchlists": int(watchlists),
            "subjects": int(subjects),
            "templates": int(templates),
            "sources": int(sources),
            "events": int(events),
        }
