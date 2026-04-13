"""
Gemini LLM initialization and configuration.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
