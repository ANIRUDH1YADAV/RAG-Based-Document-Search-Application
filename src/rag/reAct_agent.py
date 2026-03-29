"""
Agent setup for document retrieval and question answering.
"""

import os

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import Config
from src.llms.groq import llm
from src.rag.retriever_setup import get_retriever

config = Config()

tools = [get_retriever()]

if os.path.exists("description.txt"):
    with open("description.txt", "r", encoding="utf-8") as f:
        description = f.read()
else:
    description = None

prompt = ChatPromptTemplate.from_messages([
    ("system", config.prompt("system_prompt")),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

react_agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    handle_parsing_errors=True,
    max_iterations=5,
    max_execution_time=60,
    verbose=True,
    return_intermediate_steps=True
)