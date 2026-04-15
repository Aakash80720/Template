"""Knowledge Graph schema constants — node labels, edge types, required properties."""
from __future__ import annotations

from models.entities import EntityType
from models.relationships import RelationType

# Node labels used in NetworkX node attributes and Neo4j labels
NODE_LABELS: set[str] = {e.value for e in EntityType}

# Edge types
EDGE_TYPES: set[str] = {r.value for r in RelationType}

# Required properties per node label (used for validation before insertion)
REQUIRED_NODE_PROPS: dict[str, list[str]] = {
    EntityType.DOCUMENT: ["id", "name", "content"],
    EntityType.CODE: ["id", "name", "file_path", "language"],
    EntityType.PROJECT: ["id", "name"],
    EntityType.CLIENT: ["id", "name"],
    EntityType.SKILL: ["id", "name"],
    EntityType.PERSON: ["id", "name"],
}

# Required properties per edge type
REQUIRED_EDGE_PROPS: dict[str, list[str]] = {
    rel_type: ["source_id", "target_id", "relation_type"]
    for rel_type in EDGE_TYPES
}
