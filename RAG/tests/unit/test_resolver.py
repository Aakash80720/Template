"""Unit tests for the entity resolver."""
import pytest
from graph.resolver import resolve_entities


def test_exact_name_dedup():
    entities = [
        {"id": "a1", "entity_type": "SKILL", "name": "Python", "aliases": []},
        {"id": "a2", "entity_type": "SKILL", "name": "python", "aliases": []},
    ]
    resolved, decisions = resolve_entities(entities)
    assert len(resolved) == 1
    assert len(decisions) == 1


def test_alias_dedup():
    entities = [
        {"id": "b1", "entity_type": "PROJECT", "name": "Alpha", "aliases": ["project-alpha"]},
        {"id": "b2", "entity_type": "PROJECT", "name": "project-alpha", "aliases": []},
    ]
    resolved, decisions = resolve_entities(entities)
    assert len(resolved) == 1


def test_different_types_not_merged():
    entities = [
        {"id": "c1", "entity_type": "SKILL", "name": "Python", "aliases": []},
        {"id": "c2", "entity_type": "PROJECT", "name": "Python", "aliases": []},
    ]
    resolved, decisions = resolve_entities(entities)
    assert len(resolved) == 2
    assert len(decisions) == 0
