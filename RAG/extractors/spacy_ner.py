"""spaCy NER extractor — extracts named entities from plain text."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from monitoring.logger import get_logger

logger = get_logger(__name__)

# Lazy-load model to avoid startup cost when not needed
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        from config.settings import settings
        try:
            _nlp = spacy.load(settings.spacy_model)
        except OSError:
            logger.warning(
                f"spaCy model '{settings.spacy_model}' not found. "
                "Falling back to 'en_core_web_sm'. "
                f"Install with: python -m spacy download {settings.spacy_model}"
            )
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# Mapping spaCy labels → our entity types
SPACY_LABEL_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "CLIENT",
    "GPE": "CLIENT",       # treat geo-political as potential client
    "PRODUCT": "PROJECT",
    "WORK_OF_ART": "DOCUMENT",
    "LAW": "DOCUMENT",
    "LANGUAGE": "SKILL",
    "NORP": "CLIENT",
}


class SpacyEntity(BaseModel):
    text: str
    label: str                   # original spaCy label
    entity_type: str             # mapped entity type
    start_char: int
    end_char: int
    sentence: str = ""


def extract_entities(text: str) -> list[SpacyEntity]:
    """Run spaCy NER on `text` and return mapped SpacyEntity objects."""
    nlp = _get_nlp()
    doc = nlp(text)

    results: list[SpacyEntity] = []
    for ent in doc.ents:
        mapped = SPACY_LABEL_MAP.get(ent.label_, "DOCUMENT")
        results.append(SpacyEntity(
            text=ent.text,
            label=ent.label_,
            entity_type=mapped,
            start_char=ent.start_char,
            end_char=ent.end_char,
            sentence=ent.sent.text.strip(),
        ))

    logger.info(f"spaCy extracted {len(results)} entities")
    return results


def extract_skills_from_text(text: str) -> list[str]:
    """Heuristic: extract noun chunks likely to be technical skills."""
    nlp = _get_nlp()
    doc = nlp(text)

    skill_candidates: list[str] = []
    for chunk in doc.noun_chunks:
        # Simple heuristic: short noun chunks with proper nouns or all caps tokens
        if len(chunk) <= 4 and any(t.is_upper or t.pos_ == "PROPN" for t in chunk):
            skill_candidates.append(chunk.text.strip())

    return list(set(skill_candidates))
