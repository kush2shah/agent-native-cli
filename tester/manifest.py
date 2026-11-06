"""The manifest: every object the tester creates, written before the create.

Reset deletes only what's listed here (safety rule 2). Entries move
intended → created (id known) → deleted, or intended → failed when the create
errored and the object doesn't exist. The file is the source of truth, not graph
state, so a crash mid-run still leaves a record of what to clean up.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

INTENDED, CREATED, FAILED, DELETED = "intended", "created", "failed", "deleted"


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.entries: list[dict] = json.loads(path.read_text()) if path.exists() else []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, indent=2))
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written manifest

    def intend(self, kind: str, name: str, created_by: str) -> dict:
        with self._lock:
            entry = {"kind": kind, "name": name, "id": None, "status": INTENDED,
                     "created_by": created_by, "ts": time.time()}
            self.entries.append(entry)
            self._save()
            return entry

    def update(self, entry: dict, **fields) -> None:
        with self._lock:
            entry.update(fields)
            self._save()

    def find(self, kind: str, name: str) -> dict | None:
        """The latest entry for this object, if any."""
        for e in reversed(self.entries):
            if e["kind"] == kind and e["name"] == name:
                return e
        return None

    def live(self) -> list[dict]:
        """Entries reset still has to handle (anything not deleted or failed)."""
        return [e for e in self.entries if e["status"] in (INTENDED, CREATED)]
