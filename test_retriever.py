from src.rag.retriever_setup import get_retriever

r = get_retriever()
results = r.invoke("project names")
print(results)