"""Input/output schemas for the ingestion pipeline API boundary."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    FILESYSTEM = "filesystem"
    GITHUB = "github"
    NOTION = "notion"
    MCP = "mcp"
    RAW_TEXT = "raw_text"


class IngestionRequest(BaseModel):
    source_type: SourceType
    source_config: dict[str, Any] = Field(
        ..., description="Connector-specific config (path, repo, page_id, etc.)"
    )
    tags: list[str] = []
    project_hint: str | None = None   # optional hint for entity linking
    client_hint: str | None = None

    @field_validator("source_config")
    @classmethod
    def config_not_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("source_config must not be empty")
        return v


class IngestionResult(BaseModel):
    success: bool
    entities_created: int = 0
    relationships_created: int = 0
    errors: list[str] = []
    summary: dict[str, Any] = {}
