"""SQLite cache for every LLM call, keyed by hash(provider, model, messages, params, seed)."""
import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def call_key(**parts):
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, response TEXT, meta TEXT)"
        )
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            row = self._db.execute("SELECT response, meta FROM calls WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        return row[0], json.loads(row[1])

    def tag_stats(self):
        """{tag: [calls, cost, in_tokens, out_tokens]} over every call ever cached."""
        out = {}
        with self._lock:
            rows = self._db.execute("SELECT meta FROM calls").fetchall()
        for (m,) in rows:
            m = json.loads(m)
            s = out.setdefault(m.get("tag", "misc"), [0, 0.0, 0, 0])
            s[0] += 1
            s[1] += m.get("cost") or 0.0
            s[2] += m.get("in") or 0
            s[3] += m.get("out") or 0
        return out

    def put(self, key, response, meta):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO calls VALUES (?, ?, ?)",
                (key, response, json.dumps(meta)),
            )
