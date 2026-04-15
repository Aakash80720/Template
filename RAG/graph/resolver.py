"""Entity resolver — deduplicates and merges entities across ingestion runs.

Strategy:
1. Exact name match (case-insensitive) within same entity_type → merge
2. Alias overlap → merge
3. (Optional) embedding cosine similarity above threshold → merge
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from monitoring.logger import get_logger

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.92   # for embedding-based dedup (future)


def _normalise(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


class MergeDecision(BaseModel):
    keep_id: str
    discard_id: str
    reason: str


def resolve_entities(
    entities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[MergeDecision]]:
    """
    Deduplicate a list of raw entity dicts.

    Returns:
        resolved   — deduplicated entity list (dicts)
        decisions  — list of merge decisions taken
    """
    # Group by entity_type
    by_type: dict[str, list[dict]] = {}
    for e in entities:
        t = e.get("entity_type", "UNKNOWN")
        by_type.setdefault(t, []).append(e)

    resolved: list[dict] = []
    decisions: list[MergeDecision] = []

    for entity_type, group in by_type.items():
        canonical_map: dict[str, dict] = {}  # norm_name → entity dict

        for entity in group:
            norm = _normalise(entity.get("name", ""))
            if not norm:
                resolved.append(entity)
                continue

            # Check aliases too
            all_norms = [norm] + [_normalise(a) for a in entity.get("aliases", [])]

            existing_key = next(
                (k for k in all_norms if k in canonical_map), None
            )

            if existing_key:
                existing = canonical_map[existing_key]
                merged = _merge(existing, entity)
                canonical_map[existing_key] = merged
                decisions.append(MergeDecision(
                    keep_id=existing["id"],
                    discard_id=entity["id"],
                    reason=f"name match: '{norm}' == '{existing_key}'",
                ))
                logger.debug(
                    f"Merged {entity_type} '{entity.get('name')}' "
                    f"into '{existing.get('name')}'"
                )
            else:
                canonical_map[norm] = entity

        resolved.extend(canonical_map.values())

    logger.info(
        f"Resolver: {len(entities)} entities → {len(resolved)} after dedup "
        f"({len(decisions)} merges)"
    )
    return resolved, decisions


def _merge(primary: dict, secondary: dict) -> dict:
    """Merge secondary into primary — primary wins on conflicts."""
    merged = {**secondary, **primary}

    # Union list fields
    for list_field in ("aliases", "tags", "skill_ids", "project_ids",
                       "document_ids", "client_ids"):
        a = primary.get(list_field, [])
        b = secondary.get(list_field, [])
        merged[list_field] = list(dict.fromkeys(a + b))

    # Merge attributes dict
    merged["attributes"] = {
        **secondary.get("attributes", {}),
        **primary.get("attributes", {}),
    }

    # Keep highest confidence
    merged["confidence"] = max(
        primary.get("confidence", 1.0),
        secondary.get("confidence", 1.0),
    )

    return merged
