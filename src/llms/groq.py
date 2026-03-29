"""
Groq LLM initialization and configuration.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

llm = ChatGroq(model="llama-3.1-8b-instant")