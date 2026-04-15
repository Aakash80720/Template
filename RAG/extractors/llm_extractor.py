"""LLM-based entity + relationship extractor with fallback chain."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from services.llm import get_extraction_llm, get_fallback_llm
from monitoring.logger import get_logger

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "config" / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def _build_entity_chain():
    prompt = ChatPromptTemplate.from_template(_load_prompt("entity_extraction.txt"))
    primary = prompt | get_extraction_llm() | StrOutputParser()
    fallback = prompt | get_fallback_llm() | StrOutputParser()
    return primary.with_fallbacks([fallback])


def _build_relationship_chain():
    prompt = ChatPromptTemplate.from_template(_load_prompt("relationship_extraction.txt"))
    primary = prompt | get_extraction_llm() | StrOutputParser()
    fallback = prompt | get_fallback_llm() | StrOutputParser()
    return primary.with_fallbacks([fallback])


# Lazy singletons
_entity_chain = None
_relationship_chain = None


def _get_entity_chain():
    global _entity_chain
    if _entity_chain is None:
        _entity_chain = _build_entity_chain()
    return _entity_chain


def _get_relationship_chain():
    global _relationship_chain
    if _relationship_chain is None:
        _relationship_chain = _build_relationship_chain()
    return _relationship_chain


def extract_entities_llm(text: str) -> list[dict[str, Any]]:
    """Run LLM entity extraction on `text`. Returns list of raw entity dicts."""
    chain = _get_entity_chain()
    try:
        raw = chain.invoke({"text": text[:8000]})  # guard context window
        data = json.loads(raw)
        entities: list[dict] = data.get("entities", data) if isinstance(data, dict) else data
        logger.info(f"LLM extracted {len(entities)} entities")
        return entities
    except Exception as exc:
        logger.error(f"LLM entity extraction failed: {exc}")
        return []


def extract_relationships_llm(
    text: str,
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run LLM relationship extraction. Returns list of raw relationship dicts."""
    chain = _get_relationship_chain()
    try:
        raw = chain.invoke({
            "text": text[:6000],
            "entities_json": json.dumps(entities, indent=2)[:2000],
        })
        rels = json.loads(raw)
        result = rels if isinstance(rels, list) else rels.get("relationships", [])
        logger.info(f"LLM extracted {len(result)} relationships")
        return result
    except Exception as exc:
        logger.error(f"LLM relationship extraction failed: {exc}")
        return []
