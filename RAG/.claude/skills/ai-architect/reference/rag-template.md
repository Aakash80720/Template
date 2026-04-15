# RAG Pipeline Template

## services/embeddings.py
```python
from langchain_openai import OpenAIEmbeddings
from config.settings import settings

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
```

## services/vectorstore.py
```python
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from services.embeddings import get_embeddings

PERSIST_DIR = "./chroma_db"

def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )

def get_retriever(k: int = 4) -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
```

## pipelines/rag/chain.py
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from services.llm import get_llm, get_fallback_llm
from services.vectorstore import get_retriever
from monitoring.logger import get_logger

logger = get_logger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
    fallback_llm = get_fallback_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm.with_fallbacks([fallback_llm])
        | StrOutputParser()
    )

    logger.info("RAG chain built successfully")
    return chain
```

## models/inputs.py (RAG)
```python
from pydantic import BaseModel, Field

class RAGQuery(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    top_k: int = Field(4, ge=1, le=20, description="Number of docs to retrieve")
    user_id: str | None = None

class RAGResponse(BaseModel):
    answer: str
    source_documents: list[str] = []
    model_used: str
```

## Data Ingestion Pattern (pipelines/rag/ingest.py)
```python
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.vectorstore import get_vectorstore
from monitoring.logger import get_logger

logger = get_logger(__name__)

def ingest_documents(source_dir: str, glob: str = "**/*.txt") -> int:
    loader = DirectoryLoader(source_dir, glob=glob, loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    logger.info(f"Ingested {len(chunks)} chunks from {len(docs)} documents")
    return len(chunks)
```
