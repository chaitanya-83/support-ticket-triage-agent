import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# LLM_BACKEND: "groq" (free tier) | "openrouter" | "hf" (needs HF credits) | "ollama" (local, free)
KEY_VARS = {"hf": "HUGGINGFACEHUB_API_TOKEN", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def backend() -> str:
    return os.getenv("LLM_BACKEND", "groq").lower()


def llm_enabled() -> bool:
    if os.getenv("USE_LLM", "true").lower() != "true":
        return False
    b = backend()
    if b == "ollama":
        return True
    return bool(os.getenv(KEY_VARS.get(b, "")))


@lru_cache(maxsize=1)
def get_llm():
    b = backend()
    if b == "groq":
        from langchain_groq import ChatGroq
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
        return ChatGroq(model=model, temperature=0, max_tokens=1024, **extra)
    if b == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
                          base_url="https://openrouter.ai/api/v1",
                          api_key=os.getenv("OPENROUTER_API_KEY"),
                          temperature=0, max_tokens=256)
    if b == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"), temperature=0)
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    endpoint = HuggingFaceEndpoint(
        repo_id=os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        task="text-generation", max_new_tokens=256, temperature=0.01,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    return ChatHuggingFace(llm=endpoint)
