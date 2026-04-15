# Canonical Code Patterns

## config/settings.py — Pydantic BaseSettings
```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model_name: str = Field("gpt-4o", env="MODEL_NAME")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    temperature: float = Field(0.0, env="TEMPERATURE")
    max_tokens: int = Field(2048, env="MAX_TOKENS")
    langsmith_api_key: str | None = Field(None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field("default", env="LANGSMITH_PROJECT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

## services/llm.py — Decoupled LLM Factory
```python
from langchain_openai import ChatOpenAI
from config.settings import settings

def get_llm(temperature: float | None = None, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model_name,
        temperature=temperature if temperature is not None else settings.temperature,
        max_tokens=settings.max_tokens,
        streaming=streaming,
        api_key=settings.openai_api_key,
    )

def get_fallback_llm() -> ChatOpenAI:
    """Cheaper/faster fallback model for non-critical paths."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=settings.openai_api_key,
    )
```

## models/state.py — LangGraph State with Pydantic
```python
from typing import Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages

class AgentState(BaseModel):
    messages: Annotated[list, add_messages]
    context: list[str] = []
    retrieved_docs: list[str] = []
    requires_human_review: bool = False
    final_answer: str | None = None

    class Config:
        arbitrary_types_allowed = True
```

## Fallback Chain Pattern
```python
from langchain_core.prompts import ChatPromptTemplate
from services.llm import get_llm, get_fallback_llm

def build_chain_with_fallback(prompt: ChatPromptTemplate):
    primary = prompt | get_llm()
    fallback = prompt | get_fallback_llm()
    return primary.with_fallbacks([fallback])
```

## guards/input_guard.py — Input Validation
```python
from pydantic import BaseModel, field_validator
from langchain_openai import OpenAIModerationChain
from config.settings import settings

class PipelineInput(BaseModel):
    query: str
    user_id: str | None = None

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

moderation_chain = OpenAIModerationChain(
    openai_api_key=settings.openai_api_key,
    error=True,  # raises on policy violation
)
```

## monitoring/logger.py — Structured Logging + LangSmith
```python
import logging
import os
from config.settings import settings

def setup_tracing():
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(name)s", "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

setup_tracing()
```

## evals/harness.py — Eval Harness Skeleton
```python
from dataclasses import dataclass
from typing import Callable
from pydantic import BaseModel

class EvalCase(BaseModel):
    input: str
    expected_output: str | None = None
    tags: list[str] = []

@dataclass
class EvalResult:
    input: str
    output: str
    passed: bool
    score: float
    notes: str = ""

def run_eval(
    pipeline_fn: Callable[[str], str],
    cases: list[EvalCase],
    scorer: Callable[[str, str | None], tuple[bool, float]] | None = None,
) -> list[EvalResult]:
    results = []
    for case in cases:
        output = pipeline_fn(case.input)
        passed, score = scorer(output, case.expected_output) if scorer else (True, 1.0)
        results.append(EvalResult(input=case.input, output=output, passed=passed, score=score))
    return results
```
