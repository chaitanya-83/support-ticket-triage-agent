"""Run once after adding your token to .env:  python -m scripts.check_llm"""
from app.llm import llm_enabled, get_llm

if not llm_enabled():
    raise SystemExit("LLM disabled: set HUGGINGFACEHUB_API_TOKEN in .env and USE_LLM=true")

print("Calling model...")
out = get_llm().invoke("Reply with exactly one word: ready")
print("Model replied:", out.content)
