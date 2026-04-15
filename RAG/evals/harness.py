"""Eval harness — run the ingestion pipeline against gold test cases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from evals.metrics import ExtractionMetrics, compute_entity_metrics
from monitoring.logger import get_logger

logger = get_logger(__name__)


class EvalCase(BaseModel):
    """A single test case for the extraction pipeline."""
    id: str
    source_type: str
    source_config: dict[str, Any]
    gold_entities: list[str] = []                  # expected entity names
    gold_relationships: list[tuple[str, str, str]] = []   # (src, rel, tgt)
    tags: list[str] = []


@dataclass
class EvalResult:
    case_id: str
    metrics: ExtractionMetrics
    entities_found: list[str]
    errors: list[str]

    @property
    def passed(self) -> bool:
        return self.metrics.f1 >= 0.7 and not self.errors


def run_eval(cases: list[EvalCase]) -> list[EvalResult]:
    """
    Run the ingestion pipeline on each eval case and compute metrics.

    Usage:
        from evals.harness import run_eval, EvalCase
        results = run_eval([
            EvalCase(
                id="test-01",
                source_type="filesystem",
                source_config={"root_path": "./tests/fixtures/docs"},
                gold_entities=["Project Alpha", "Python", "FastAPI"],
            )
        ])
        for r in results:
            print(r.case_id, r.metrics.f1, "PASS" if r.passed else "FAIL")
    """
    from pipelines.ingestion.graph import run_ingestion

    results: list[EvalResult] = []

    for case in cases:
        logger.info(f"Running eval case: {case.id}")
        errors: list[str] = []
        entities_found: list[str] = []

        try:
            state = run_ingestion(
                source_type=case.source_type,
                source_config=case.source_config,
                thread_id=f"eval-{case.id}",
            )
            entities_found = [
                e.get("name", "")
                for e in state.get("embedded_entities", [])
            ]
        except Exception as exc:
            errors.append(str(exc))
            logger.error(f"Eval case {case.id} failed: {exc}")

        metrics = compute_entity_metrics(entities_found, case.gold_entities)
        results.append(EvalResult(
            case_id=case.id,
            metrics=metrics,
            entities_found=entities_found,
            errors=errors,
        ))

    passed = sum(1 for r in results if r.passed)
    logger.info(f"Eval complete: {passed}/{len(results)} cases passed")
    return results
