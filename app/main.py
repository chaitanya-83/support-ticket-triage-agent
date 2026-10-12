from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from . import db
from .graph import build_graph
from .llm import backend, llm_enabled
from .schemas import TicketIn, TicketOut
import os

graph = None
PAGE = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    db.init_db()
    graph = build_graph()
    yield


app = FastAPI(title="Support Ticket Triage Agent", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return PAGE.read_text(encoding="utf-8")


@app.get("/info")
def info():
    b = backend()
    model = os.getenv({"groq": "GROQ_MODEL", "hf": "HF_MODEL", "ollama": "OLLAMA_MODEL",
                       "openrouter": "OPENROUTER_MODEL"}.get(b, "GROQ_MODEL"), "")
    return {"llm_enabled": llm_enabled(), "backend": b, "model": model}


@app.post("/tickets", response_model=TicketOut)
def create_ticket(ticket: TicketIn):
    result = graph.invoke({"text": ticket.text, "customer": ticket.customer, "trace": []})
    return TicketOut(
        ticket_id=result["ticket_id"],
        category=result["category"],
        priority=result["priority"],
        assigned_agent=result["assigned_agent"],
        assignment_reason=result["assignment_reason"],
        trace=result["trace"],
    )


@app.get("/agents")
def agents():
    return db.get_agent_workloads()


@app.get("/tickets")
def tickets(limit: int = 50):
    return db.list_tickets(limit)


@app.post("/reset")
def reset():
    db.reset_db()
    return {"status": "demo data reset"}
