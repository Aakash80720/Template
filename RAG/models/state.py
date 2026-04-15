"""LangGraph pipeline state — Pydantic model with annotated message list."""
from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from models.entities import AnyEntity, BaseEntity
from models.relationships import Relationship


class IngestionState(BaseModel):
    """State flowing through the 6-node knowledge-graph ingestion pipeline."""

    # LangGraph message bus (append-only)
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Raw payloads from connectors (one item per ingested source)
    raw_payloads: list[dict[str, Any]] = Field(default_factory=list)

    # Entities extracted in extract_node; refined in resolve_node
    extracted_entities: list[dict[str, Any]] = Field(default_factory=list)
    resolved_entities: list[dict[str, Any]] = Field(default_factory=list)

    # Relationships extracted in link_node
    relationships: list[dict[str, Any]] = Field(default_factory=list)

    # Entities with embeddings attached (set in embed_node)
    embedded_entities: list[dict[str, Any]] = Field(default_factory=list)

    # Pipeline control
    current_source: str | None = None       # connector id of active source
    iteration_count: int = 0
    requires_human_review: bool = False
    review_reason: str | None = None

    # Final summary written by store_node
    ingestion_summary: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
