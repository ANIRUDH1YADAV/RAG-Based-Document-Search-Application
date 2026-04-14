"""
API routes for RAG operations.
"""

import logging
import re

from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from src.core.config import settings
from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

router = APIRouter()
logger = logging.getLogger(__name__)

NOISY_SNIPPET_MARKERS = (
    "retriever tool instruction",
    "important note",
    "disclaimer",
    "usage:",
    "upload a resume",
    "enter a question",
    "this retriever tool",
)


def _looks_like_education_query(query: str) -> bool:
    """Return True when query intent is likely about education details."""
    lowered = query.lower()
    return any(
        key in lowered
        for key in (
            "education",
            "degree",
            "qualification",
            "school",
            "college",
            "university",
            "cgpa",
            "percentage",
            "student",
            "academic",
        )
    )


def _extract_education_lines(snippets: list[str], max_lines: int = 3) -> list[str]:
    """Extract concise education-related lines from retrieved snippets."""
    def _clean_institute_text(value: str) -> str:
        cleaned = re.sub(r"\S+@\S+", "", value)
        cleaned = re.sub(r"\+?\d[\d\s-]{7,}", "", cleaned)
        cleaned = re.sub(r"\b(linkedin|github)\S*", "", cleaned, flags=re.IGNORECASE)
        cleaned = " ".join(cleaned.split()).strip(" -–,")

        marker = "netaji subhas university of technology"
        lowered = cleaned.lower()
        if marker in lowered:
            cleaned = cleaned[lowered.index(marker):]

        return cleaned

    joined_text = " ".join(" ".join(snippet.split()) for snippet in snippets)

    btech_table_match = re.search(
        r"CGPA/Percentage Year B\.?Tech\s+(?P<institute>.*?)\s+"
        r"(?P<score>\d+(?:\.\d+)?)\s+(?P<year>20\d{2}\s*[–-]\s*20\d{2})",
        joined_text,
        flags=re.IGNORECASE,
    )

    btech_match = re.search(
        r"B\.?Tech\s+(?P<institute>.*?)\s+(?P<score>\d+(?:\.\d+)?)\s+"
        r"(?P<year>20\d{2}\s*[–-]\s*20\d{2})",
        joined_text,
        flags=re.IGNORECASE,
    )
    senior_match = re.search(
        r"Senior Secondary\s+(?P<institute>.*?)\s+"
        r"(?P<score>\d+(?:\.\d+)?%)\s+(?P<year>20\d{2})",
        joined_text,
        flags=re.IGNORECASE,
    )
    secondary_match = re.search(
        r"(?<!Senior )Secondary\s+(?P<institute>.*?)\s+"
        r"(?P<score>\d+(?:\.\d+)?%)\s+(?P<year>20\d{2})",
        joined_text,
        flags=re.IGNORECASE,
    )

    structured_lines: list[str] = []
    btech_data = btech_table_match or btech_match
    if btech_data:
        institute = _clean_institute_text(btech_data.group("institute"))
        structured_lines.append(
            "B.Tech - "
            f"{institute} - "
            f"CGPA {btech_data.group('score').strip()} - "
            f"{btech_data.group('year').strip()}"
        )
    if senior_match:
        institute = _clean_institute_text(senior_match.group("institute"))
        structured_lines.append(
            "Senior Secondary - "
            f"{institute} - "
            f"{senior_match.group('score').strip()} - "
            f"{senior_match.group('year').strip()}"
        )
    if secondary_match:
        institute = _clean_institute_text(secondary_match.group("institute"))
        structured_lines.append(
            "Secondary - "
            f"{institute} - "
            f"{secondary_match.group('score').strip()} - "
            f"{secondary_match.group('year').strip()}"
        )

    if structured_lines:
        return structured_lines[:max_lines]

    extracted: list[str] = []
    seen = set()

    for snippet in snippets:
        for part in re.split(r"(?<=[.!?])\s+|\s{2,}", snippet):
            candidate = " ".join(part.split())
            lowered = candidate.lower()
            if not candidate:
                continue
            if any(marker in lowered for marker in NOISY_SNIPPET_MARKERS):
                continue
            if not any(
                token in lowered
                for token in (
                    "education",
                    "b.tech",
                    "btech",
                    "degree",
                    "university",
                    "school",
                    "secondary",
                    "cgpa",
                    "percentage",
                )
            ):
                continue

            if candidate in seen:
                continue
            seen.add(candidate)
            extracted.append(candidate)
            if len(extracted) >= max_lines:
                return extracted

    return extracted


def _retrieve_qdrant_fallback_snippets(query: str, k: int = 3) -> list[str]:
    """Retrieve snippets directly from Qdrant payloads for no-LLM fallback."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        points, _ = client.scroll(
            collection_name=settings.CODE_COLLECTION,
            limit=250,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.warning("Qdrant fallback retrieval failed: %s", exc)
        return []

    terms = [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if len(token) > 2
    ]
    if _looks_like_education_query(query):
        terms.extend(
            [
                "education",
                "degree",
                "certificate",
                "university",
                "college",
                "school",
                "cgpa",
                "percentage",
                "secondary",
                "btech",
                "b.tech",
            ]
        )

    scored: list[tuple[int, str]] = []
    for point in points:
        payload = point.payload or {}
        text = str(payload.get("page_content") or payload.get("text") or "")
        normalized_text = " ".join(text.split())
        if not normalized_text:
            continue

        lowered = normalized_text.lower()
        if any(marker in lowered for marker in NOISY_SNIPPET_MARKERS):
            continue
        score = sum(lowered.count(term) for term in terms)
        if terms and score == 0:
            continue

        scored.append((score, normalized_text[:500]))

    if not scored:
        for point in points[:k]:
            payload = point.payload or {}
            text = str(payload.get("page_content") or payload.get("text") or "")
            normalized_text = " ".join(text.split())
            if normalized_text:
                scored.append((0, normalized_text[:500]))

    scored.sort(key=lambda item: item[0], reverse=True)

    snippets: list[str] = []
    seen = set()
    for _, snippet in scored:
        if snippet in seen:
            continue
        snippets.append(snippet)
        seen.add(snippet)
        if len(snippets) >= k:
            break

    return snippets


@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """
    Process a RAG query and return the result.
    """
    chat_history = ChatHistory.get_session_history(req.session_id)
    try:
        await chat_history.add_message(HumanMessage(content=req.query))

        messages = await chat_history.get_messages()
        result = builder.invoke({
            "messages": messages
        })
        output_text = result["messages"][-1].content

        await chat_history.add_message(AIMessage(content=output_text))
        return {"result": {"content": output_text}}

    except Exception as exc:
        logger.exception("RAG query failed for session %s", req.session_id)
        snippets = _retrieve_qdrant_fallback_snippets(req.query, k=3)
        if snippets:
            if _looks_like_education_query(req.query):
                education_lines = _extract_education_lines(snippets, max_lines=3)
                if education_lines:
                    formatted = "\n".join(
                        f"{index + 1}. {line}"
                        for index, line in enumerate(education_lines)
                    )
                    fallback_text = (
                        "The model service is currently unavailable, but based on your uploaded "
                        f"document I found these education details:\n\n{formatted}"
                    )
                else:
                    snippet_lines = "\n\n".join(
                        f"{index + 1}. {snippet}"
                        for index, snippet in enumerate(snippets)
                    )
                    fallback_text = (
                        "The model service is currently unavailable, but I found relevant content "
                        "from your uploaded documents:\n\n"
                        f"{snippet_lines}"
                    )
            else:
                snippet_lines = "\n\n".join(
                    f"{index + 1}. {snippet}"
                    for index, snippet in enumerate(snippets)
                )
                fallback_text = (
                    "The model service is currently unavailable, but I found relevant content "
                    "from your uploaded documents:\n\n"
                    f"{snippet_lines}"
                )
        else:
            fallback_text = (
                "I could not process your query right now because the model service is unavailable. "
                "Please check GOOGLE_API_KEY, model access, and quota limits."
            )
        try:
            await chat_history.add_message(AIMessage(content=fallback_text))
        except Exception:
            logger.exception(
                "Failed to persist fallback assistant message for session %s",
                req.session_id,
            )

        return {
            "result": {
                "content": fallback_text,
                "error": str(exc),
            }
        }


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description")
):
    try:
        status_upload = documents(description, file)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Document upload failed: {e}")

    if not status_upload:
        raise HTTPException(
            status_code=500,
            detail="Failed to store document chunks in Qdrant.",
        )

    return {"status": True}