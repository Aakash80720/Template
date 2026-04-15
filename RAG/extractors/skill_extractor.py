"""Rule-based skill extractor — pattern matching against a known skill taxonomy.

Combines:
1. A curated SKILL_TAXONOMY dict of canonical names + aliases
2. spaCy noun-chunk heuristics for unknown skills
3. tree-sitter import analysis for code-derived skills
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from monitoring.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Skill taxonomy — extend this dict for your domain
# ---------------------------------------------------------------------------

SKILL_TAXONOMY: dict[str, list[str]] = {
    # Languages
    "Python": ["python", "py", ".py"],
    "TypeScript": ["typescript", "ts", ".ts"],
    "JavaScript": ["javascript", "js", ".js", "node.js", "nodejs"],
    "Go": ["golang", "go lang"],
    "Rust": ["rust-lang"],
    "Java": ["java 17", "java 21"],
    # Frameworks
    "FastAPI": ["fast api", "fastapi"],
    "Django": ["django rest", "drf"],
    "React": ["reactjs", "react.js"],
    "Next.js": ["nextjs", "next js"],
    "LangChain": ["langchain"],
    "LangGraph": ["langgraph"],
    "LlamaIndex": ["llama-index", "llamaindex"],
    # Data / ML
    "PyTorch": ["torch", "pytorch"],
    "TensorFlow": ["tensorflow", "tf"],
    "scikit-learn": ["sklearn"],
    "Pandas": ["pandas", "pd"],
    # Infrastructure
    "Docker": ["dockerfile", "docker-compose"],
    "Kubernetes": ["k8s", "kubectl"],
    "Terraform": ["tf files", "hcl"],
    "PostgreSQL": ["postgres", "psql"],
    "Redis": ["redis cache"],
    "Elasticsearch": ["elastic", "opensearch"],
    # AI / LLM
    "OpenAI API": ["openai", "gpt-4", "gpt-4o", "chatgpt"],
    "Anthropic Claude": ["claude", "claude-3"],
    "Hugging Face": ["huggingface", "hf transformers", "transformers"],
    "spaCy": ["spacy"],
    "ChromaDB": ["chroma", "chromadb"],
    "Pinecone": ["pinecone"],
}

# Build reverse lookup: alias (lower) → canonical name
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in SKILL_TAXONOMY.items():
    _ALIAS_TO_CANONICAL[canonical.lower()] = canonical
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


class ExtractedSkill(BaseModel):
    canonical_name: str
    matched_text: str
    source: str   # "taxonomy" | "spacy" | "import"
    confidence: float = 1.0


def match_skills_from_text(text: str) -> list[ExtractedSkill]:
    """Match text against the skill taxonomy using substring search."""
    text_lower = text.lower()
    results: list[ExtractedSkill] = []
    seen: set[str] = set()

    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        # Use word-boundary regex to avoid partial matches
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, text_lower) and canonical not in seen:
            results.append(ExtractedSkill(
                canonical_name=canonical,
                matched_text=alias,
                source="taxonomy",
                confidence=0.95,
            ))
            seen.add(canonical)

    return results


def match_skills_from_imports(imports: list[str]) -> list[ExtractedSkill]:
    """Infer skills from parsed import statements."""
    results: list[ExtractedSkill] = []
    seen: set[str] = set()

    for imp in imports:
        imp_lower = imp.lower()
        for alias, canonical in _ALIAS_TO_CANONICAL.items():
            if alias in imp_lower and canonical not in seen:
                results.append(ExtractedSkill(
                    canonical_name=canonical,
                    matched_text=imp[:80],
                    source="import",
                    confidence=0.9,
                ))
                seen.add(canonical)

    return results


def extract_skills(
    text: str,
    imports: list[str] | None = None,
) -> list[ExtractedSkill]:
    """Combined skill extraction from text + imports."""
    skills = match_skills_from_text(text)
    if imports:
        skills += match_skills_from_imports(imports)

    # Deduplicate by canonical name
    seen: set[str] = set()
    unique: list[ExtractedSkill] = []
    for s in skills:
        if s.canonical_name not in seen:
            unique.append(s)
            seen.add(s.canonical_name)

    logger.info(f"SkillExtractor found {len(unique)} skills")
    return unique
