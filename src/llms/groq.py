"""
Gemini LLM initialization and configuration.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
configured_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash").strip()

# Some API versions reject *-latest aliases, so normalize to a stable model id.
if configured_model.endswith("-latest"):
	configured_model = configured_model.removesuffix("-latest")

os.environ["GOOGLE_API_KEY"] = google_api_key

llm = ChatGoogleGenerativeAI(
	model=configured_model or "gemini-2.0-flash",
	request_timeout=20,
	retries=2,
)
