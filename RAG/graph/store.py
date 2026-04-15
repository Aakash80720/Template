"""Graph persistence layer — NetworkX (default) with optional Neo4j backend.

Usage:
    store = GraphStore()          # uses NetworkX + JSON file
    store = GraphStore(neo4j=True) # uses Neo4j (requires NEO4J_* env vars)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from config.settings import settings
from graph.builder import KnowledgeGraphBuilder
from models.entities import BaseEntity
from models.relationships import Relationship
from monitoring.logger import get_logger

logger = get_logger(__name__)


class GraphStore:
    def __init__(self, neo4j: bool = False) -> None:
        self._neo4j_enabled = neo4j and bool(settings.neo4j_uri)
        self.builder = KnowledgeGraphBuilder()
        self._persist_path = Path(settings.kg_persist_path)

        if self._neo4j_enabled:
            self._init_neo4j()
        else:
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_entities(self, entities: list[BaseEntity]) -> None:
        self.builder.add_entities(entities)
        if self._neo4j_enabled:
            self._neo4j_upsert_entities(entities)

    def upsert_relationships(self, rels: list[Relationship]) -> None:
        self.builder.add_relationships(rels)
        if self._neo4j_enabled:
            self._neo4j_upsert_relationships(rels)

    def save(self) -> None:
        """Persist the NetworkX graph to disk as JSON."""
        if self._neo4j_enabled:
            return  # Neo4j is already persistent
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(self.builder.graph)
        self._persist_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Graph saved to {self._persist_path}")

    def stats(self) -> dict[str, Any]:
        return self.builder.stats()

    # ------------------------------------------------------------------
    # NetworkX persistence
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        if self._persist_path.exists():
            try:
                data = json.loads(self._persist_path.read_text())
                self.builder.graph = json_graph.node_link_graph(data, directed=True, multigraph=True)
                logger.info(
                    f"Graph loaded from {self._persist_path}: "
                    f"{self.builder.graph.number_of_nodes()} nodes, "
                    f"{self.builder.graph.number_of_edges()} edges"
                )
            except Exception as exc:
                logger.warning(f"Could not load graph from disk: {exc}")

    # ------------------------------------------------------------------
    # Neo4j backend (stub — wire up with neo4j Python driver)
    # ------------------------------------------------------------------

    def _init_neo4j(self) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            logger.info("Neo4j driver initialised")
        except ImportError as exc:
            raise ImportError("Install neo4j: pip install neo4j") from exc

    def _neo4j_upsert_entities(self, entities: list[BaseEntity]) -> None:
        # TODO: implement MERGE-based upsert per entity_type
        # Example pattern:
        #   MERGE (n:Document {id: $id})
        #   ON CREATE SET n += $props
        #   ON MATCH SET n += $props
        raise NotImplementedError("Neo4j entity upsert not yet implemented")

    def _neo4j_upsert_relationships(self, rels: list[Relationship]) -> None:
        # TODO: implement MERGE-based relationship upsert
        # Example pattern:
        #   MATCH (a {id: $source_id}), (b {id: $target_id})
        #   MERGE (a)-[r:REL_TYPE {id: $id}]->(b)
        #   ON CREATE SET r += $props
        raise NotImplementedError("Neo4j relationship upsert not yet implemented")
