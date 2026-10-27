from __future__ import annotations

import sqlite3
from pathlib import Path

from oculi.db import Database


def test_prune_and_backup(tmp_path: Path) -> None:
    db = Database(tmp_path / "ops.db")
    try:
        db.add_event({"type": "face", "source_id": "cam", "ts": "2026-01-01T00:00:00Z"})
        db.add_event({"type": "object", "source_id": "cam", "ts": "2026-01-01T00:00:01Z"})
        assert db.counts()["events"] == 2

        removed = db.prune_events(0)
        assert removed == 2
        assert db.counts()["events"] == 0

        db.add_event({"type": "plate", "source_id": "cam", "ts": "2026-01-01T00:00:02Z"})
        backup = Path(db.backup(tmp_path / "backup.db"))
        assert backup.exists()
        conn = sqlite3.connect(backup)
        try:
            assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        finally:
            conn.close()
    finally:
        db.close()
