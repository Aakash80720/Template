"""LangGraph node functions for the 6-stage ingestion pipeline.

Nodes (in order):
    1. ingest_node    — fetch payloads from connectors
    2. extract_node   — spaCy NER + tree-sitter + LLM extraction
    3. resolve_node   — entity deduplication (HITL checkpoint here)
    4. link_node      — relationship extraction
    5. embed_node     — compute + attach embeddings
    6. store_node     — persist to KG + vector store
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from config.settings import settings
from connectors.mcp_bridge import CONNECTOR_REGISTRY
from extractors.llm_extractor import extract_entities_llm, extract_relationships_llm
from extractors.skill_extractor import extract_skills
from extractors.spacy_ner import extract_entities as spacy_extract
from extractors.tree_sitter_parser import parse_code
from graph.resolver import resolve_entities
from graph.store import GraphStore
from models.entities import (
    ENTITY_CLASS_MAP,
    CodeEntity,
    DocumentEntity,
    EntityType,
    SkillEntity,
)
from models.relationships import Relationship, RelationType
from models.state import IngestionState
from monitoring.logger import get_logger
from services.embeddings import get_embeddings
from services.vectorstore import upsert_entity_texts

logger = get_logger(__name__)

# Shared graph store (singleton for the pipeline run)
_store: GraphStore | None = None


def _get_store() -> GraphStore:
    global _store
    if _store is None:
        _store = GraphStore()
    return _store


# ---------------------------------------------------------------------------
# Node 1: ingest_node
# ---------------------------------------------------------------------------

def ingest_node(state: IngestionState) -> dict:
    """Fetch raw content from the registered connector for the active source."""
    source_type = state.current_source
    if not source_type:
        return {"errors": state.errors + ["ingest_node: no current_source set"]}

    connector = CONNECTOR_REGISTRY.get(source_type)
    if not connector:
        return {"errors": state.errors + [f"Unknown connector: {source_type}"]}

    # config is expected in the last human message as JSON
    try:
        config_msg = next(
            (m for m in reversed(state.messages) if isinstance(m, HumanMessage)), None
        )
        config: dict = json.loads(config_msg.content) if config_msg else {}
    except Exception:
        config = {}

    payloads = connector.fetch(config)
    raw = [p.model_dump() for p in payloads]

    logger.info(f"ingest_node: {len(raw)} payloads from '{source_type}'")
    return {
        "raw_payloads": raw,
        "messages": [AIMessage(content=f"Ingested {len(raw)} payloads from {source_type}")],
        "iteration_count": state.iteration_count + 1,
    }


# ---------------------------------------------------------------------------
# Node 2: extract_node
# ---------------------------------------------------------------------------

def extract_node(state: IngestionState) -> dict:
    """Run spaCy NER + tree-sitter + LLM extraction on all raw payloads."""
    all_entities: list[dict] = []

    for payload in state.raw_payloads:
        content: str = payload.get("raw_content", "")
        meta: dict = payload.get("metadata", {})
        is_code: bool = meta.get("is_code", False)

        # ---- Code path: tree-sitter ----------------------------------------
        if is_code:
            file_path = meta.get("file_path") or payload.get("uri", "unknown")
            parsed = parse_code(content, file_path)
            if parsed:
                skills = extract_skills(content, parsed.imports)
                for sk in skills:
                    all_entities.append({
                        "entity_type": EntityType.SKILL,
                        "name": sk.canonical_name,
                        "confidence": sk.confidence,
                        "attributes": {"source": "import"},
                    })
                all_entities.append({
                    "entity_type": EntityType.CODE,
                    "name": file_path,
                    "file_path": file_path,
                    "language": parsed.language,
                    "content": content[:4000],  # truncate for storage
                    "functions": parsed.functions,
                    "classes": parsed.classes,
                    "imports": parsed.imports,
                })
            continue

        # ---- Document path: spaCy NER + LLM ---------------------------------
        spacy_ents = spacy_extract(content)
        for e in spacy_ents:
            all_entities.append({
                "entity_type": e.entity_type,
                "name": e.text,
                "confidence": 0.8,
                "attributes": {"context": e.sentence},
            })

        llm_ents = extract_entities_llm(content)
        all_entities.extend(llm_ents)

        # Always create a Document entity for the payload itself
        all_entities.append({
            "entity_type": EntityType.DOCUMENT,
            "name": meta.get("file_name") or payload.get("uri", "unnamed"),
            "content": content[:6000],
            "file_path": payload.get("uri", ""),
            "tags": [],
        })

    logger.info(f"extract_node: {len(all_entities)} raw entities")
    return {
        "extracted_entities": all_entities,
        "messages": [AIMessage(content=f"Extracted {len(all_entities)} entities")],
    }


# ---------------------------------------------------------------------------
# Node 3: resolve_node  (HITL checkpoint)
# ---------------------------------------------------------------------------

def resolve_node(state: IngestionState) -> dict:
    """Deduplicate entities. If HITL enabled and merge count is high, pause."""
    resolved, decisions = resolve_entities(state.extracted_entities)

    if settings.hitl_enabled and len(decisions) > 10:
        # Pause here — LangGraph will resume when the caller calls .update_state()
        human_decision = interrupt({
            "question": "Many entity merges were made. Approve to continue?",
            "merges": [d.model_dump() for d in decisions[:5]],  # preview first 5
            "total_merges": len(decisions),
        })
        logger.info(f"HITL resume decision: {human_decision}")

    logger.info(f"resolve_node: {len(state.extracted_entities)} → {len(resolved)}")
    return {
        "resolved_entities": resolved,
        "messages": [AIMessage(content=f"Resolved to {len(resolved)} unique entities")],
    }


# ---------------------------------------------------------------------------
# Node 4: link_node
# ---------------------------------------------------------------------------

def link_node(state: IngestionState) -> dict:
    """Extract relationships between resolved entities using LLM."""
    # Build a short text summary of all entity names for the LLM
    entity_names_text = "\n".join(
        f"{e.get('entity_type')}: {e.get('name')}"
        for e in state.resolved_entities[:50]  # cap context
    )

    raw_rels = extract_relationships_llm(
        text=entity_names_text,
        entities=state.resolved_entities[:50],
    )

    logger.info(f"link_node: {len(raw_rels)} raw relationships")
    return {
        "relationships": raw_rels,
        "messages": [AIMessage(content=f"Linked {len(raw_rels)} relationships")],
    }


# ---------------------------------------------------------------------------
# Node 5: embed_node
# ---------------------------------------------------------------------------

def embed_node(state: IngestionState) -> dict:
    """Compute embeddings for each resolved entity and attach to entity dict."""
    embeddings_model = get_embeddings()
    entities = state.resolved_entities

    texts = [
        f"{e.get('entity_type')} {e.get('name')} {e.get('content', '')[:500]}"
        for e in entities
    ]

    try:
        vectors = embeddings_model.embed_documents(texts)
    except Exception as exc:
        logger.error(f"Embedding failed: {exc}")
        vectors = [[] for _ in entities]

    embedded = []
    for entity, vec in zip(entities, vectors):
        embedded.append({**entity, "embedding": vec})

    logger.info(f"embed_node: embedded {len(embedded)} entities")
    return {
        "embedded_entities": embedded,
        "messages": [AIMessage(content=f"Embedded {len(embedded)} entities")],
    }


# ---------------------------------------------------------------------------
# Node 6: store_node
# ---------------------------------------------------------------------------

def store_node(state: IngestionState) -> dict:
    """Persist entities + relationships to KG and vector store."""
    store = _get_store()

    # Materialise Pydantic entity objects
    entity_objects = []
    for raw in state.embedded_entities:
        etype = raw.get("entity_type")
        cls = ENTITY_CLASS_MAP.get(etype)
        if cls is None:
            continue
        try:
            obj = cls.model_validate(raw)
            entity_objects.append(obj)
        except Exception as exc:
            logger.warning(f"Entity validation failed: {exc} — {raw.get('name')}")

    store.upsert_entities(entity_objects)

    # Materialise relationship objects
    rel_objects = []
    for raw in state.relationships:
        try:
            rel = Relationship.model_validate(raw)
            rel_objects.append(rel)
        except Exception as exc:
            logger.warning(f"Relationship validation failed: {exc}")

    store.upsert_relationships(rel_objects)
    store.save()

    # Upsert into vector store for semantic search
    ids = [e.id for e in entity_objects]
    texts = [f"{e.entity_type} {e.name}" for e in entity_objects]
    metas = [{"entity_type": e.entity_type, "name": e.name} for e in entity_objects]
    if ids:
        upsert_entity_texts(ids, texts, metas)

    stats = store.stats()
    logger.info(f"store_node: {stats}")

    return {
        "ingestion_summary": {
            "entities_stored": len(entity_objects),
            "relationships_stored": len(rel_objects),
            "graph_stats": stats,
        },
        "messages": [AIMessage(content=f"Stored {len(entity_objects)} entities, {len(rel_objects)} relationships")],
    }
