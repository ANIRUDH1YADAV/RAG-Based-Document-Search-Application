"""
API client for communicating with backend services.
"""

import logging
import requests

logger = logging.getLogger(__name__)

PYTHON_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 20
UPLOAD_TIMEOUT_SECONDS = 180


def query_backend(query: str, session_id: str) -> str:
    """
    Send a query to the RAG backend.

    Args:
        query: The user's query text.
        session_id: Session identifier for tracking conversation.

    Returns:
        Response text from the backend or error message.
    """
    url = f"{PYTHON_BASE_URL}/rag/query"
    print(f"[query_backend] Calling: {url}")

    try:
        response = requests.post(
            url,
            json={"query": query, "session_id": session_id},
            allow_redirects=False,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        logger.error("Backend query request failed: %s", exc)
        return "Error: Unable to reach backend API at http://127.0.0.1:8000. Is uvicorn running?"

    if response.status_code == 200:
        return response.json()["result"]["content"]

    return f"Error: {response.status_code} - {response.text}"


def document_upload_rag(file, description: str) -> tuple[bool, str | None]:
    """
    Upload a document to the RAG system.

    Args:
        file: File object to upload.
        description: Description of the document.

    Returns:
        Tuple where first value indicates success and second value is
        an optional error message.
    """
    headers = {
        "X-Description": description
    }
    url = f"{PYTHON_BASE_URL}/rag/documents/upload"

    if not file:
        return False, "No file provided for upload."

    file_mime_type = file.type or "application/octet-stream"
    files = {"file": (file.name, file, file_mime_type)}
    try:
        response = requests.post(
            url,
            files=files,
            headers=headers,
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        logger.error(
            "Document upload timed out after %s seconds.",
            UPLOAD_TIMEOUT_SECONDS,
        )
        return (
            False,
            f"Upload timed out after {UPLOAD_TIMEOUT_SECONDS} seconds. "
            "Please try again.",
        )
    except requests.exceptions.RequestException as exc:
        logger.error("Document upload request failed: %s", exc)
        return (
            False,
            "Unable to reach backend API at http://127.0.0.1:8000. "
            "Is uvicorn running?",
        )

    logger.info("Upload API response status: %s", response.status_code)

    if response.status_code != 200:
        return (
            False,
            f"Upload failed with status {response.status_code}.",
        )

    try:
        payload = response.json()
    except ValueError:
        return False, "Upload failed: backend returned an invalid response."

    if payload.get("status") is True:
        return True, None

    detail = payload.get("detail") or payload.get("error") or payload.get("message")
    if detail:
        return False, f"Upload failed: {detail}"

    return False, "Upload failed. Check backend logs for details."