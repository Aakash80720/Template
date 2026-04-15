"""Unit tests for the skill extractor."""
from extractors.skill_extractor import extract_skills, match_skills_from_imports


def test_taxonomy_match():
    skills = extract_skills("We use FastAPI and PostgreSQL for this service.")
    names = [s.canonical_name for s in skills]
    assert "FastAPI" in names
    assert "PostgreSQL" in names


def test_import_match():
    imports = ["from langchain_openai import ChatOpenAI", "import spacy"]
    skills = match_skills_from_imports(imports)
    names = [s.canonical_name for s in skills]
    assert any("LangChain" in n or "OpenAI" in n for n in names)
    assert "spaCy" in names


def test_no_false_positives():
    skills = extract_skills("The weather is nice today.")
    assert len(skills) == 0
