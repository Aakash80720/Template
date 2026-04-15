"""Pydantic entity models for all Knowledge Graph node types."""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    DOCUMENT = "DOCUMENT"
    CODE = "CODE"
    PROJECT = "PROJECT"
    CLIENT = "CLIENT"
    SKILL = "SKILL"
    PERSON = "PERSON"


class CodeLanguage(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Base entity
# ---------------------------------------------------------------------------

class BaseEntity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType
    name: str
    aliases: list[str] = []
    attributes: dict[str, Any] = {}
    embedding: list[float] | None = None   # populated by embed_node
    source_connector: str | None = None    # which connector produced this
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Concrete entity types
# ---------------------------------------------------------------------------

class DocumentEntity(BaseEntity):
    entity_type: EntityType = EntityType.DOCUMENT
    title: str
    content: str
    url: str | None = None
    file_path: str | None = None
    mime_type: str = "text/plain"
    language: str = "en"
    tags: list[str] = []
    author_ids: list[str] = []       # → PersonEntity ids
    project_ids: list[str] = []      # → ProjectEntity ids


class CodeEntity(BaseEntity):
    entity_type: EntityType = EntityType.CODE
    file_path: str
    language: CodeLanguage = CodeLanguage.UNKNOWN
    content: str
    functions: list[str] = []        # extracted by tree-sitter
    classes: list[str] = []
    imports: list[str] = []
    dependencies: list[str] = []
    repository: str | None = None
    commit_sha: str | None = None
    skill_ids: list[str] = []        # → SkillEntity ids


class ProjectEntity(BaseEntity):
    entity_type: EntityType = EntityType.PROJECT
    description: str = ""
    tech_stack: list[str] = []
    repository_url: str | None = None
    status: str = "active"           # active | archived | planning
    client_ids: list[str] = []       # → ClientEntity ids
    document_ids: list[str] = []     # → DocumentEntity ids
    skill_ids: list[str] = []        # → SkillEntity ids


class ClientEntity(BaseEntity):
    entity_type: EntityType = EntityType.CLIENT
    industry: str = ""
    website: str | None = None
    contact_emails: list[str] = []
    project_ids: list[str] = []      # → ProjectEntity ids


class SkillEntity(BaseEntity):
    entity_type: EntityType = EntityType.SKILL
    category: str = ""               # e.g. "framework", "language", "domain"
    level: str = "intermediate"      # beginner | intermediate | expert
    related_skill_ids: list[str] = []


class PersonEntity(BaseEntity):
    entity_type: EntityType = EntityType.PERSON
    email: str | None = None
    github_handle: str | None = None
    project_ids: list[str] = []
    skill_ids: list[str] = []


# ---------------------------------------------------------------------------
# Union helper
# ---------------------------------------------------------------------------

AnyEntity = (
    DocumentEntity
    | CodeEntity
    | ProjectEntity
    | ClientEntity
    | SkillEntity
    | PersonEntity
)

ENTITY_CLASS_MAP: dict[EntityType, type[BaseEntity]] = {
    EntityType.DOCUMENT: DocumentEntity,
    EntityType.CODE: CodeEntity,
    EntityType.PROJECT: ProjectEntity,
    EntityType.CLIENT: ClientEntity,
    EntityType.SKILL: SkillEntity,
    EntityType.PERSON: PersonEntity,
}
