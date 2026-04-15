"""Vector store factory — stores entity embeddings for semantic search."""
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import settings
from services.embeddings import get_embeddings


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=get_embeddings(),
    )


def get_retriever(k: int = 5) -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3},
    )


def upsert_entity_texts(
    entity_ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:
    """Add or update entity text representations in the vector store."""
    vs = get_vectorstore()
    vs.add_texts(texts=texts, metadatas=metadatas, ids=entity_ids)
