"""
Llama 3 LLM initialization and configuration (via Ollama).
"""
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")
