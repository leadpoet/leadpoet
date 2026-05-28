"""SQLite cache with TTL. Thread-safe enough for our single-process async use."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


class Cache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        # Each call uses its own connection (works fine for async via run_in_executor-style use,
        # but we just use it synchronously since SQLite ops are sub-ms).
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def make_key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> Optional[Any]:
        conn = self._conn()
        try:
            cur = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key=?", (key,)
            )
            row = cur.fetchone()
            if not row:
                return None
            value, expires_at = row
            if expires_at < int(time.time()):
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        finally:
            conn.close()

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        now = int(time.time())
        conn = self._conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (key, json.dumps(value, default=str), now + ttl_seconds, now),
            )
            conn.commit()
        finally:
            conn.close()

    def stats(self) -> dict:
        conn = self._conn()
        try:
            now = int(time.time())
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            live = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE expires_at > ?", (now,)
            ).fetchone()[0]
            return {"total": total, "live": live, "expired": total - live}
        finally:
            conn.close()
