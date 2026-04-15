"""Evaluation metrics for the knowledge-graph extraction pipeline."""
from __future__ import annotations

from pydantic import BaseModel


class ExtractionMetrics(BaseModel):
    """Precision / recall / F1 over entity extraction against a gold set."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def compute_entity_metrics(
    predicted: list[str],
    gold: list[str],
) -> ExtractionMetrics:
    """
    Compare predicted entity names to gold entity names (case-insensitive).

    Args:
        predicted: list of entity names returned by the pipeline
        gold:      list of expected entity names from the gold set
    """
    pred_set = {n.lower().strip() for n in predicted}
    gold_set = {n.lower().strip() for n in gold}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    return ExtractionMetrics(true_positives=tp, false_positives=fp, false_negatives=fn)


def compute_relationship_accuracy(
    predicted: list[tuple[str, str, str]],   # (source, rel_type, target)
    gold: list[tuple[str, str, str]],
) -> float:
    """Fraction of predicted (src, rel, tgt) triples that appear in gold."""
    if not predicted:
        return 0.0
    pred_set = {(s.lower(), r.lower(), t.lower()) for s, r, t in predicted}
    gold_set = {(s.lower(), r.lower(), t.lower()) for s, r, t in gold}
    return len(pred_set & gold_set) / len(pred_set)
