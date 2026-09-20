"""Deadline and durable per-round journals for reproducible experiments."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DEADLINE = "2026-09-14T07:00:00+08:00"


class ExperimentPaused(RuntimeError):
    """The user's wall-clock deadline was reached; partial data is not training data."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_deadline(value: str | None) -> datetime | None:
    if value is None:
        return None
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("experiment deadline must specify a timezone")
    return result.astimezone(timezone.utc)


def seconds_remaining(deadline: str | datetime | None) -> float:
    if deadline is None:
        return float("inf")
    when = parse_deadline(deadline) if isinstance(deadline, str) else deadline
    return (when - utc_now()).total_seconds()


def check_deadline(deadline: str | datetime | None, *, reserve_seconds: float = 0) -> None:
    if seconds_remaining(deadline) <= reserve_seconds:
        raise ExperimentPaused("Paused at the user-specified deadline; completed rounds are journaled.")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def fingerprint(value: Any) -> str:
    return sha256(json_bytes(value)).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    with pending.open("wb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


class RunJournal:
    """Each completed interval is durable before another solve starts.

    A crash during an append may leave an incomplete last line. Recovery retains
    only complete records; the last interval is rerun from its prior state.
    A changed scenario, model, method or physical configuration cannot be resumed.
    """
    def __init__(self, directory: str | Path, identity: dict):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.digest = fingerprint(identity)
        self.header = self.directory / "initial.json"
        self.round_path = self.directory / "rounds.jsonl"
        self.status_path = self.directory / "status.json"

    def initialize(self, initial_state: dict, plans: dict) -> None:
        if self.header.exists():
            existing = json.loads(self.header.read_text(encoding="utf-8"))
            if existing["fingerprint"] != self.digest:
                raise ValueError("refusing to resume an experiment with changed inputs or model")
            return
        if self.round_path.exists() and self.round_path.stat().st_size:
            raise ValueError("round journal exists without its initial state")
        atomic_json(self.header, {"fingerprint": self.digest, "identity": self.identity,
                                  "initial_state": initial_state, "dayahead_plan": plans})

    def recover(self) -> tuple[dict, dict, list[dict]]:
        header = json.loads(self.header.read_text(encoding="utf-8"))
        if header["fingerprint"] != self.digest:
            raise ValueError("refusing to resume an experiment with changed inputs or model")
        records = []
        if self.round_path.exists():
            # Truncation repairs only this run's incomplete append, never a valid record.
            with self.round_path.open("r+b") as stream:
                valid_end = 0
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        stream.truncate(valid_end)
                        break
                    record = json.loads(line)
                    if record["period"] != len(records):
                        raise ValueError("non-consecutive periods in experiment journal")
                    records.append(record)
                    valid_end = stream.tell()
        return header["initial_state"], header["dayahead_plan"], records

    def append(self, record: dict) -> None:
        with self.round_path.open("ab") as stream:
            stream.write(json_bytes(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.status("running", completed_periods=record["period"] + 1)

    def status(self, state: str, **extra) -> None:
        atomic_json(self.status_path, {"state": state, "updated_at": utc_now().isoformat(),
                                      "fingerprint": self.digest, **extra})
