"""Knowledge Graph builder — converts extracted entities and relationships
into a NetworkX MultiDiGraph, with optional Neo4j sync."""
from __future__ import annotations

from typing import Any

import networkx as nx

from models.entities import BaseEntity
from models.relationships import Relationship
from graph.schema import REQUIRED_NODE_PROPS, REQUIRED_EDGE_PROPS
from monitoring.logger import get_logger

logger = get_logger(__name__)


class KnowledgeGraphBuilder:
    """
    Builds and maintains an in-memory NetworkX MultiDiGraph.
    Each node stores the full entity dict as attributes.
    Each edge stores the full relationship dict as attributes.
    """

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_entity(self, entity: BaseEntity) -> None:
        data = entity.model_dump()
        self._validate_node(data)
        self.graph.add_node(entity.id, **data)
        logger.debug(f"Node added: {entity.entity_type} '{entity.name}' ({entity.id})")

    def add_entities(self, entities: list[BaseEntity]) -> None:
        for e in entities:
            self.add_entity(e)

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        if self.graph.has_node(entity_id):
            return dict(self.graph.nodes[entity_id])
        return None

    def update_entity(self, entity_id: str, updates: dict[str, Any]) -> None:
        if self.graph.has_node(entity_id):
            self.graph.nodes[entity_id].update(updates)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_relationship(self, rel: Relationship) -> None:
        data = rel.model_dump()
        self._validate_edge(data)
        self.graph.add_edge(
            rel.source_id,
            rel.target_id,
            key=rel.id,
            **data,
        )
        logger.debug(
            f"Edge added: {rel.source_id} --[{rel.relation_type}]--> {rel.target_id}"
        )

    def add_relationships(self, rels: list[Relationship]) -> None:
        for r in rels:
            self.add_relationship(r)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: str | None = None,
        direction: str = "out",   # "out" | "in" | "both"
    ) -> list[dict[str, Any]]:
        if not self.graph.has_node(entity_id):
            return []

        if direction == "out":
            edges = self.graph.out_edges(entity_id, data=True, keys=True)
            neighbor_ids = [v for _, v, _, d in edges
                            if relation_type is None or d.get("relation_type") == relation_type]
        elif direction == "in":
            edges = self.graph.in_edges(entity_id, data=True, keys=True)
            neighbor_ids = [u for u, _, _, d in edges
                            if relation_type is None or d.get("relation_type") == relation_type]
        else:
            out_ids = self.get_neighbors(entity_id, relation_type, "out")
            in_ids = self.get_neighbors(entity_id, relation_type, "in")
            return out_ids + in_ids

        return [self.get_entity(nid) for nid in neighbor_ids if self.get_entity(nid)]

    def subgraph_by_type(self, entity_type: str) -> nx.MultiDiGraph:
        nodes = [n for n, d in self.graph.nodes(data=True)
                 if d.get("entity_type") == entity_type]
        return self.graph.subgraph(nodes).copy()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("entity_type", "UNKNOWN")
            type_counts[t] = type_counts.get(t, 0) + 1

        edge_counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            r = data.get("relation_type", "UNKNOWN")
            edge_counts[r] = edge_counts.get(r, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": type_counts,
            "edges_by_type": edge_counts,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_node(self, data: dict) -> None:
        entity_type = data.get("entity_type", "")
        required = REQUIRED_NODE_PROPS.get(entity_type, ["id", "name"])
        missing = [k for k in required if not data.get(k)]
        if missing:
            logger.warning(f"Node missing required props: {missing} (type={entity_type})")

    def _validate_edge(self, data: dict) -> None:
        rel_type = data.get("relation_type", "")
        required = REQUIRED_EDGE_PROPS.get(rel_type, ["source_id", "target_id"])
        missing = [k for k in required if not data.get(k)]
        if missing:
            logger.warning(f"Edge missing required props: {missing} (type={rel_type})")
