"""
Retriever setup and vector store configuration.
"""

import os

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore

from src.core.config import settings

# ✅ (NO CHANGE) - Embeddings config is correct
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_API_KEY")
)


def retriever_chain(chunks: list[Document]):
    """
    Initialize and store documents in Qdrant vector database.
    """

    try:
        # ✅ (NO CHANGE) - This already creates collection automatically
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


def get_retriever():
    """
    Get a retriever tool connected to the Qdrant vector store.
    """

    try:
        # 🔥 CHANGE 1: Wrapped in try block (previously no safety)
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name=settings.CODE_COLLECTION,
        )

        print("✅ Using existing Qdrant collection")

    except Exception:
        # 🔥 CHANGE 2: HANDLE COLLECTION NOT FOUND
        print("⚠️ Collection not found → creating new one")

        dummy_doc = Document(
            page_content="Initialization document",
            metadata={"source": "init"}
        )

        # 🔥 CHANGE 3: CREATE COLLECTION AUTOMATICALLY
        vectorstore = QdrantVectorStore.from_documents(
            documents=[dummy_doc],
            embedding=embeddings,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name=settings.CODE_COLLECTION,
        )

        print("✅ New collection created")

    # ✅ (NO CHANGE) - convert to retriever
    retriever = vectorstore.as_retriever()

    # 🔥 CHANGE 4: SAFE DESCRIPTION HANDLING
    if os.path.exists("description.txt"):
        with open("description.txt", "r", encoding="utf-8") as f:
            description = f.read()
    else:
        description = "user uploaded documents"   # ← fallback added

    # 🔥 CHANGE 5: BETTER TOOL DESCRIPTION (important for LLM)
    retriever_tool = create_retriever_tool(
        retriever,
        "retriever_customer_uploaded_documents",
        f"Use this tool ONLY to answer questions about: {description}. "
        "Do NOT use it for general questions."
    )

    return retriever_tool