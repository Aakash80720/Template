"""Metric tracking for extraction quality drift detection.

Records per-run stats to a JSONL file. Compare runs over time to catch
extraction quality regressions (e.g. entity count drops, confidence drifts).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from monitoring.logger import get_logger

logger = get_logger(__name__)

DRIFT_LOG_PATH = Path("./data/drift_log.jsonl")


class RunMetrics:
    """Collect and persist metrics for a single ingestion run."""

    def __init__(self, run_id: str, source_type: str) -> None:
        self.run_id = run_id
        self.source_type = source_type
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._data: dict[str, Any] = {}

    def record(self, key: str, value: Any) -> None:
        self._data[key] = value

    def flush(self) -> None:
        """Append metrics to the drift log file."""
        DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "run_id": self.run_id,
            "source_type": self.source_type,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            **self._data,
        }
        with DRIFT_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Drift metrics flushed for run {self.run_id}")


def load_drift_history(last_n: int = 100) -> list[dict[str, Any]]:
    """Load the last N run metric entries for comparison."""
    if not DRIFT_LOG_PATH.exists():
        return []
    lines = DRIFT_LOG_PATH.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-last_n:]]


def check_entity_count_drift(current: int, history: list[dict]) -> bool:
    """
    Returns True if the current entity count is more than 50% below the
    rolling average of previous runs — a simple drift signal.
    """
    if not history:
        return False
    avg = sum(h.get("entities_extracted", 0) for h in history) / len(history)
    if avg > 0 and current < avg * 0.5:
        logger.warning(
            f"Drift detected: current entity count ({current}) is "
            f"more than 50% below rolling average ({avg:.1f})"
        )
        return True
    return False
