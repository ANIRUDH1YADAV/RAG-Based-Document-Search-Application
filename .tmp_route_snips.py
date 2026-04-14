from src.api.routes import retrieve_context_snippets
snips = retrieve_context_snippets("education of the student?", k=3)
print(len(snips))
for i, s in enumerate(snips, 1):
    print(f"[{i}] {s[:200]}")
