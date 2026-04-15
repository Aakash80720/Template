from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model_name: str = Field("gpt-4o", env="MODEL_NAME")
    fallback_model_name: str = Field("gpt-4o-mini", env="FALLBACK_MODEL_NAME")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    temperature: float = Field(0.0, env="TEMPERATURE")
    max_tokens: int = Field(4096, env="MAX_TOKENS")

    # LangSmith
    langsmith_api_key: str | None = Field(None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field("knowledge-graph", env="LANGSMITH_PROJECT")

    # Vector store
    chroma_persist_dir: str = Field("./data/chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field("kg_entities", env="CHROMA_COLLECTION")

    # Knowledge graph persistence
    kg_persist_path: str = Field("./data/knowledge_graph.json", env="KG_PERSIST_PATH")

    # Neo4j (optional — swap for NetworkX when not set)
    neo4j_uri: str | None = Field(None, env="NEO4J_URI")
    neo4j_user: str | None = Field(None, env="NEO4J_USER")
    neo4j_password: str | None = Field(None, env="NEO4J_PASSWORD")

    # MCP bridge
    mcp_server_url: str | None = Field(None, env="MCP_SERVER_URL")
    mcp_auth_token: str | None = Field(None, env="MCP_AUTH_TOKEN")

    # GitHub connector
    github_token: str | None = Field(None, env="GITHUB_TOKEN")

    # spaCy
    spacy_model: str = Field("en_core_web_trf", env="SPACY_MODEL")

    # Pipeline behaviour
    hitl_enabled: bool = Field(True, env="HITL_ENABLED")
    max_iterations: int = Field(20, env="MAX_ITERATIONS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
