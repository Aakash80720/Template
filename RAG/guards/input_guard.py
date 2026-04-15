"""Input validation guard — validates and sanitises ingestion requests."""
from __future__ import annotations

from pydantic import BaseModel, field_validator

from models.inputs import IngestionRequest
from monitoring.logger import get_logger

logger = get_logger(__name__)

MAX_CONTENT_LENGTH = 1_000_000  # 1 MB of raw text per payload


class GuardedIngestionRequest(IngestionRequest):
    """Extends IngestionRequest with additional safety checks."""

    @field_validator("source_config")
    @classmethod
    def no_path_traversal(cls, v: dict) -> dict:
        path = str(v.get("root_path", "") or v.get("path", ""))
        if ".." in path:
            raise ValueError("Path traversal detected in source_config")
        return v


def validate_payload_content(content: str) -> str:
    """Strip null bytes; reject oversized payloads."""
    content = content.replace("\x00", "")
    if len(content) > MAX_CONTENT_LENGTH:
        logger.warning(f"Payload truncated from {len(content)} to {MAX_CONTENT_LENGTH} chars")
        content = content[:MAX_CONTENT_LENGTH]
    return content


def validate_request(raw: dict) -> GuardedIngestionRequest:
    """Parse and validate an ingestion request dict. Raises ValidationError on failure."""
    request = GuardedIngestionRequest.model_validate(raw)
    logger.info(f"Input guard passed for source_type='{request.source_type}'")
    return request
