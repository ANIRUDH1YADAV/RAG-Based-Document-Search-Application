"""
Retriever setup and vector store configuration.
"""

import os

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.core.config import settings

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Global variable to store the Qdrant vectorstore instance
# This ensures get_retriever() can access documents stored by retriever_chain()
_qdrant_vectorstore = None


def _get_qdrant_client() -> QdrantClient:
    if not settings.QDRANT_URL:
        raise ValueError("QDRANT_URL is not set. Configure it in your .env file.")
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


def _ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(settings.CODE_COLLECTION):
        return

    vector_size = len(embeddings.embed_query("collection-init"))
    client.create_collection(
        collection_name=settings.CODE_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def retriever_chain(chunks: list[Document]):
    """
    Initialize and store documents in Qdrant vector database.

    Args:
        chunks: List of document chunks to store.

    Returns:
        Boolean indicating success of the operation.
    """
    global _qdrant_vectorstore

    try:
        client = _get_qdrant_client()
        _ensure_collection(client)
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=settings.CODE_COLLECTION,
            embedding=embeddings,
        )
        vectorstore.add_documents(chunks)

        # Store the vectorstore globally so get_retriever() can access it
        _qdrant_vectorstore = vectorstore

        print("Qdrant vector store initialized with documents")
        print(f"Vectorstore contains {len(chunks)} document chunks")
        return True
    except Exception as e:
        print(f"Error storing documents in Qdrant: {e}")
        return False


def get_retriever():
    """
    Get a retriever tool connected to the Qdrant vector store.

    Returns the retriever tool that can search documents stored by retriever_chain().
    If no documents have been uploaded yet, uses an empty Qdrant collection.

    Returns:
        A LangChain retriever tool configured for the vector store.

    Raises:
        Exception: If vector store initialization fails.
    """
    global _qdrant_vectorstore

    try:
        # Use the global vectorstore if it exists (documents have been uploaded)
        if _qdrant_vectorstore is not None:
            retriever = _qdrant_vectorstore.as_retriever()
            print("Using existing Qdrant vectorstore with uploaded documents")
        else:
            print("No documents uploaded yet, using empty Qdrant collection")
            client = _get_qdrant_client()
            _ensure_collection(client)
            _qdrant_vectorstore = QdrantVectorStore(
                client=client,
                collection_name=settings.CODE_COLLECTION,
                embedding=embeddings,
            )
            retriever = _qdrant_vectorstore.as_retriever()

        # Load document description
        if os.path.exists("description.txt"):
            with open("description.txt", "r", encoding="utf-8") as f:
                description = f.read()
        else:
            description = None

        retriever_tool = create_retriever_tool(
            retriever,
            "retriever_customer_uploaded_documents",
            f"Use this tool **only** to answer questions about: {description}\n"
            "Don't use this tool to answer anything else."
        )

        return retriever_tool

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise Exception(e)
