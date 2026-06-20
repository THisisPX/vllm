# SPDX-License-Identifier: Apache-2.0
"""Thread-safe JSONL writer for expert trace records."""

import json
import os
import threading
from pathlib import Path


class TraceWriter:
    """Appends JSON records to a file, one JSON object per line.

    This is intentionally simple: no async I/O, no compression.
    For the experimental scale (tens of thousands of records per
    request) this is sufficient.
    """

    def __init__(self, path: Path, flush_interval: int = 10_000) -> None:
        """Create a writer.

        Args:
            path: Path to the output ``.jsonl`` file.
            flush_interval: Number of records between automatic
                ``f.flush()`` calls.
        """
        self._path = path
        self._flush_interval = flush_interval
        self._count = 0
        self._lock = threading.Lock()

        # Ensure parent directory exists.
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8")

    def write(self, record: dict) -> None:
        """Append a single trace record as a JSON line."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._file.write(line)
            self._count += 1
            if self._count % self._flush_interval == 0:
                self._file.flush()

    def write_many(self, records: list[dict]) -> None:
        """Append multiple records in one batch."""
        if not records:
            return
        lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        lines += "\n"
        with self._lock:
            self._file.write(lines)
            self._count += len(records)
            if self._count % self._flush_interval < len(records):
                self._file.flush()

    def flush(self) -> None:
        """Force-flush buffered data to disk."""
        with self._lock:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        """Flush and close the file."""
        with self._lock:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()

    @property
    def record_count(self) -> int:
        return self._count
