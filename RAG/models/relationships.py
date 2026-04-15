"""Pydantic relationship (edge) models for the Knowledge Graph."""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RelationType(str, Enum):
    OWNS_PROJECT = "OWNS_PROJECT"           # Client → Project
    HAS_DOCUMENT = "HAS_DOCUMENT"           # Project → Document
    USES_SKILL = "USES_SKILL"               # Project → Skill
    IMPLEMENTS_SKILL = "IMPLEMENTS_SKILL"   # Code → Skill
    REFERENCES = "REFERENCES"               # Document → Code | Document
    AUTHORED_BY = "AUTHORED_BY"             # Document/Code → Person
    WORKED_ON = "WORKED_ON"                 # Person → Project
    REQUIRES_SKILL = "REQUIRES_SKILL"       # Project/Client → Skill
    RELATED_TO = "RELATED_TO"              # generic semantic similarity


class Relationship(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    relation_type: RelationType
    attributes: dict[str, Any] = {}
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    weight: float = Field(1.0, ge=0.0)     # for graph traversal algorithms

    class Config:
        use_enum_values = True
