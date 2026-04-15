"""Output validation guard — validates pipeline results before returning to caller."""
from __future__ import annotations

from models.inputs import IngestionResult
from monitoring.logger import get_logger

logger = get_logger(__name__)

MIN_ENTITIES_WARN_THRESHOLD = 1   # warn if fewer than this many entities were stored


def validate_result(raw: dict) -> IngestionResult:
    """Parse and validate the ingestion result. Adds warnings for suspicious output."""
    result = IngestionResult.model_validate(raw)

    if result.success and result.entities_created < MIN_ENTITIES_WARN_THRESHOLD:
        logger.warning(
            "Output guard: pipeline succeeded but created 0 entities. "
            "Check extractor output and source content."
        )

    if result.errors:
        logger.warning(f"Output guard: pipeline finished with errors: {result.errors}")

    return result
