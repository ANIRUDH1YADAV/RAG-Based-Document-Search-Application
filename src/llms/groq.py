"""
Llama 3 LLM initialization and configuration (via Groq).
"""

import os

from langchain_groq import ChatGroq

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

llm = ChatGroq(model=GROQ_MODEL, temperature=GROQ_TEMPERATURE)
