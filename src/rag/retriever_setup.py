"""
Retriever setup and vector store configuration.
"""

import os
import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool, tool
from langchain_qdrant import QdrantVectorStore

from src.core.config import settings

DEFAULT_HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GOOGLE_EMBEDDING_MODEL = "models/text-embedding-004"


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Convert env-style strings to bool values."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@lru_cache(maxsize=1)
def _get_embeddings():
    """Create embeddings using endpoint or local model based on environment."""
    backend = os.getenv("EMBEDDINGS_BACKEND", "huggingface").strip().lower()
    model_name = os.getenv("HF_EMBEDDING_MODEL", DEFAULT_HF_EMBEDDING_MODEL)
    hf_api_key = os.getenv("HF_API_KEY", "").strip()
    local_files_only = _as_bool(os.getenv("HF_LOCAL_FILES_ONLY"), default=False)
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    google_model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", DEFAULT_GOOGLE_EMBEDDING_MODEL)

    if backend in {"google", "gemini"}:
        if not google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required when EMBEDDINGS_BACKEND=google")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        print("Using Google embedding model")
        return GoogleGenerativeAIEmbeddings(
            model=google_model_name,
            google_api_key=google_api_key,
        )

    if backend in {"huggingface_local", "hf_local", "local"}:
        from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

        print("Using local Hugging Face embeddings")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"local_files_only": local_files_only},
        )

    if backend not in {"huggingface", "huggingface_endpoint", "hf_endpoint", "endpoint"}:
        raise RuntimeError(
            f"Unsupported EMBEDDINGS_BACKEND '{backend}'. "
            "Use one of: huggingface, huggingface_local, google"
        )

    if not hf_api_key:
        raise RuntimeError("HF_API_KEY is required when EMBEDDINGS_BACKEND=huggingface")

    from langchain_huggingface.embeddings.huggingface_endpoint import HuggingFaceEndpointEmbeddings

    print("Using Hugging Face endpoint embeddings")
    return HuggingFaceEndpointEmbeddings(
        model=model_name,
        huggingfacehub_api_token=hf_api_key,
    )


def _get_description() -> str:
    """Read uploaded document description used by the retriever tool."""
    if os.path.exists("description.txt"):
        with open("description.txt", "r", encoding="utf-8") as file_handle:
            return file_handle.read()
    return "user uploaded documents"


def _build_unavailable_retriever_tool(error: Exception):
    """Return a fallback tool so server startup does not fail on retriever init errors."""
    error_message = f"{type(error).__name__}: {error}"

    @tool("retriever_customer_uploaded_documents")
    def retriever_customer_uploaded_documents(query: str) -> str:
        """Use this tool to answer questions from uploaded documents."""
        return (
            "Document retriever is unavailable right now. "
            "Check embedding/Qdrant configuration and restart the backend. "
            f"Details: {error_message}"
        )

    return retriever_customer_uploaded_documents


def _get_or_create_vectorstore(embeddings):
    """Get existing Qdrant collection or create it with a dummy doc."""
    try:
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name=settings.CODE_COLLECTION,
        )
        print("Using existing Qdrant collection")
        return vectorstore
    except Exception:
        print("Collection not found or incompatible. Creating a new one.")
        dummy_doc = Document(
            page_content="Initialization document",
            metadata={"source": "init"},
        )
        vectorstore = QdrantVectorStore.from_documents(
            documents=[dummy_doc],
            embedding=embeddings,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name=settings.CODE_COLLECTION,
        )
        print("New Qdrant collection created")
        return vectorstore


def retrieve_context_snippets(query: str, k: int = 3) -> list[str]:
    """Retrieve top matching document chunks for direct fallback responses."""
    try:
        embeddings = _get_embeddings()
        vectorstore = _get_or_create_vectorstore(embeddings)
        docs = vectorstore.similarity_search(query, k=k)
    except Exception as error:
        print(f"Context retrieval fallback failed: {error}")
        return []

    snippets: list[str] = []
    for doc in docs:
        text = " ".join(str(doc.page_content).split())
        if text:
            snippets.append(text[:500])

    return snippets


def retrieve_lexical_context_snippets(
    query: str,
    k: int = 3,
    scan_limit: int = 250,
) -> list[str]:
    """Retrieve fallback snippets using lexical matching over Qdrant payload text."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        points, _ = client.scroll(
            collection_name=settings.CODE_COLLECTION,
            limit=scan_limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as error:
        print(f"Lexical fallback retrieval failed: {error}")
        return []

    query_terms = [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if len(token) > 2
    ]

    scored_snippets: list[tuple[int, str]] = []
    for point in points:
        payload = point.payload or {}
        text = str(payload.get("page_content") or payload.get("text") or "")
        if not text.strip():
            continue

        normalized_text = " ".join(text.split())
        lowered = normalized_text.lower()
        score = sum(lowered.count(term) for term in query_terms)
        if query_terms and score == 0:
            continue

        scored_snippets.append((score, normalized_text[:500]))

    if not scored_snippets:
        # If no lexical match exists, return a few chunks as a generic fallback.
        generic_snippets: list[str] = []
        for point in points[:k]:
            payload = point.payload or {}
            text = str(payload.get("page_content") or payload.get("text") or "")
            normalized_text = " ".join(text.split())
            if normalized_text:
                generic_snippets.append(normalized_text[:500])
        return generic_snippets

    scored_snippets.sort(key=lambda item: item[0], reverse=True)

    unique_snippets: list[str] = []
    seen = set()
    for _, snippet in scored_snippets:
        if snippet in seen:
            continue
        unique_snippets.append(snippet)
        seen.add(snippet)
        if len(unique_snippets) >= k:
            break

    return unique_snippets


def retriever_chain(chunks: list[Document]):
    """
    Initialize and store documents in Qdrant vector database.
    """

    try:
        embeddings = _get_embeddings()
        vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name=settings.CODE_COLLECTION,
        )

        print("✅ Qdrant vector store initialized with documents")
        print(f"Vectorstore contains {len(chunks)} document chunks")

        return True

    except Exception as e:
        print(f"❌ Error storing documents in Qdrant: {e}")
        return False


@lru_cache(maxsize=1)
def get_retriever():
    """
    Get a retriever tool connected to the Qdrant vector store.
    """

    try:
        embeddings = _get_embeddings()
        vectorstore = _get_or_create_vectorstore(embeddings)

        retriever = vectorstore.as_retriever()
        description = _get_description()

        return create_retriever_tool(
            retriever,
            "retriever_customer_uploaded_documents",
            f"Use this tool ONLY to answer questions about: {description}. "
            "Do NOT use it for general questions.",
        )
    except Exception as error:
        print(f"Retriever initialization failed: {error}")
        return _build_unavailable_retriever_tool(error)
