"""
Common tools for document and description processing.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from src.llms.groq import llm

logger = logging.getLogger(__name__)

DESCRIPTION_ENHANCEMENT_TIMEOUT_SECONDS = 12


def _invoke_description_enhancer(prompt: str) -> str:
    """Invoke the LLM and normalize content to text."""
    response = llm.invoke(prompt)
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def enhance_description_with_llm(user_description: str) -> str:
    """
    Enhance user-provided document description using LLM.

    Rewrites the description to be suitable as a retriever tool instruction
    that clearly indicates the tool is only for answering questions about
    the uploaded content.

    Args:
        user_description: The original user-provided description.

    Returns:
        Enhanced description formatted as a tool instruction.
    """
    fallback_description = user_description.strip()
    if not fallback_description:
        return "This tool is for answering questions about uploaded documents."

    prompt = f"""
    Rewrite the following user-provided document description to be used as a retriever tool instruction.
    It should clearly state that the tool is only for answering questions about the uploaded content.

    Description: "{user_description}"

    Tool Instruction:"""

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_invoke_description_enhancer, prompt)
    try:
        enhanced_description = future.result(
            timeout=DESCRIPTION_ENHANCEMENT_TIMEOUT_SECONDS
        )
        return enhanced_description or fallback_description
    except TimeoutError:
        future.cancel()
        logger.warning(
            "Description enhancement timed out after %s seconds. Using user description.",
            DESCRIPTION_ENHANCEMENT_TIMEOUT_SECONDS,
        )
        return fallback_description
    except Exception as exc:
        logger.warning(
            "Description enhancement failed. Using user description. Error: %s",
            exc,
        )
        return fallback_description
    finally:
        executor.shutdown(wait=False, cancel_futures=True)