# Support Ticket Triage & Smart Assignment Agent (POC)

Single-agent LangGraph workflow:
`read ticket -> classify -> assign priority -> check workload -> assign agent`

Stack: Python, LangGraph, LangChain, Groq / OpenRouter / Hugging Face / Ollama (switchable), Pydantic, SQLite, FastAPI.

## 1. Setup
```bash
python -m venv venv
venv\Scripts\activate             # macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env             # macOS/Linux: cp .env.example .env
```

## 2. Pick a free LLM backend (edit `.env`, then run `python -m scripts.check_llm`)
| Backend | Setup | Notes |
|---|---|---|
| `groq` (default) | key from https://console.groq.com/keys -> `GROQ_API_KEY` | Free, no card; gpt-oss-20b or any model your key lists (run scripts.list_groq_models) |
| `openrouter` | key from https://openrouter.ai/keys; use a model ending in `:free` | ~50 requests/day until you buy credits |
| `hf` | HF token with credits | Free HF accounts get only ~$0.10/month |
| `ollama` | install Ollama, `ollama pull llama3.2:3b`, `pip install langchain-ollama` | Fully local |

Never commit `.env` (already in `.gitignore`). Set `USE_LLM=false` to run on the rule fallback only.

## 3. Run the live demo
Start the server and open http://127.0.0.1:8000 for the dashboard (submit tickets, watch the trace and agent workloads update live). Swagger docs are at /docs.

```bash
uvicorn app.main:app --reload
```

### API only
```bash
uvicorn app.main:app --reload
```
Swagger UI: http://127.0.0.1:8000/docs

```bash
curl -X POST http://127.0.0.1:8000/tickets -H "Content-Type: application/json" \
  -d '{"text":"I was charged twice for my order and need a refund urgently","customer":"Ravi"}'
curl http://127.0.0.1:8000/agents
```
Expected: category billing/refund, priority 1-2, assigned to **Kumar** (3 open, no high priority)
rather than **Deepak** (5 open, includes P1/P2). The response includes a step-by-step `trace`.

## 4. Test and evaluate
```bash
pytest -q                       # offline, uses the rule fallback, no token needed
python -m scripts.evaluate      # runs the 15 labelled tickets through the real LLM
```

## Assignment logic
Load score = sum over open tickets (P1=5, P2=3, P3=2, P4=1). Among agents with the matching skill,
the lowest score wins, ties broken by fewer open tickets. Agents with 8+ open tickets are skipped if
anyone else is free. To reset demo data, delete `tickets.db`.

## Project layout
```
app/        schemas.py  db.py  llm.py  graph.py  main.py
scripts/    check_llm.py  evaluate.py
tests/      test_flow.py
data/       eval_tickets.json
```

## No Hugging Face credits? Run a free local model
1. Install Ollama from https://ollama.com and run `ollama pull llama3.2:3b`
2. In `.env` set `LLM_BACKEND=ollama` and `OLLAMA_MODEL=llama3.2:3b`
3. `pip install langchain-ollama`, then `python -m scripts.check_llm`
