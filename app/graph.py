"""Single-agent, sequential LangGraph workflow:
read ticket -> classify -> assign priority -> check workload -> assign agent
"""
import re
from typing import List, TypedDict

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END

from . import db
from .llm import get_llm, llm_enabled
from .schemas import Category, Classification, PriorityResult


class TicketState(TypedDict, total=False):
    customer: str
    text: str
    category: str
    priority: int
    agents: list
    assigned_agent: str
    assignment_reason: str
    ticket_id: int
    trace: List[str]


def _clean(msg):
    """Strip <think> blocks / extra text so only the JSON object reaches the Pydantic parser."""
    text = getattr(msg, "content", msg)
    if isinstance(text, list):
        text = "".join(str(part) for part in text)
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text


CLEAN = RunnableLambda(_clean)


# ---------- prompts + Pydantic parsers ----------
cls_parser = PydanticOutputParser(pydantic_object=Classification)
CLS_PROMPT = PromptTemplate(
    template=(
        "You are a support ticket classifier. Classify the ticket into exactly one category "
        "from: billing, refund, damaged, delivery, technical, account, other.\n"
        "{format_instructions}\nReturn ONLY the JSON.\n\nTicket: {ticket}"
    ),
    input_variables=["ticket"],
    partial_variables={"format_instructions": cls_parser.get_format_instructions()},
)

pri_parser = PydanticOutputParser(pydantic_object=PriorityResult)
PRI_PROMPT = PromptTemplate(
    template=(
        "You assign priority to support tickets: 1=critical (money lost, legal threat, service "
        "down, angry repeated contact), 2=high, 3=medium, 4=low (general question).\n"
        "Category: {category}\n{format_instructions}\nReturn ONLY the JSON.\n\nTicket: {ticket}"
    ),
    input_variables=["ticket", "category"],
    partial_variables={"format_instructions": pri_parser.get_format_instructions()},
)

# ---------- offline fallbacks (also used if the LLM call/parse fails) ----------
KEYWORDS = {
    "refund": ["refund", "money back", "return"],
    "billing": ["charged", "invoice", "bill", "payment", "double"],
    "damaged": ["damaged", "broken", "cracked", "defective"],
    "delivery": ["delivery", "not arrived", "late", "tracking", "shipping"],
    "technical": ["error", "crash", "bug", "login", "not working"],
    "account": ["password", "account", "email change", "profile"],
}
URGENT = ["urgent", "asap", "immediately", "legal", "fraud", "charged twice", "lawyer", "escalate"]


def _fallback_category(text: str) -> str:
    t = text.lower()
    for cat, words in KEYWORDS.items():
        if any(w in t for w in words):
            return cat
    return "other"


def _fallback_priority(text: str) -> int:
    t = text.lower()
    hits = sum(w in t for w in URGENT)
    return 1 if hits >= 2 else 2 if hits == 1 else 3


# ---------- nodes ----------
def read_ticket(state: TicketState) -> TicketState:
    text = re.sub(r"\s+", " ", state["text"]).strip()
    return {"text": text, "trace": [f"read_ticket: {len(text)} chars received"]}


def classify(state: TicketState) -> TicketState:
    source = "llm"
    try:
        if not llm_enabled():
            raise RuntimeError("LLM disabled")
        res = (CLS_PROMPT | get_llm() | CLEAN | cls_parser).invoke({"ticket": state["text"]})
        category = res.category.value
    except Exception:
        category, source = _fallback_category(state["text"]), "keyword-fallback"
    return {"category": category, "trace": state["trace"] + [f"classify: {category} ({source})"]}


def assign_priority(state: TicketState) -> TicketState:
    source = "llm"
    try:
        if not llm_enabled():
            raise RuntimeError("LLM disabled")
        res = (PRI_PROMPT | get_llm() | CLEAN | pri_parser).invoke(
            {"ticket": state["text"], "category": state["category"]}
        )
        priority = res.priority
    except Exception:
        priority, source = _fallback_priority(state["text"]), "rule-fallback"
    # Safety net: explicit urgency words can only raise priority, never lower it
    if any(w in state["text"].lower() for w in URGENT):
        priority = min(priority, 2)
    return {"priority": priority, "trace": state["trace"] + [f"priority: P{priority} ({source})"]}


def check_workload(state: TicketState) -> TicketState:
    agents = db.get_agent_workloads()
    summary = ", ".join(
        f"{a['name']}(open={a['open_count']}, high={a['high_priority_count']}, load={a['load_score']})"
        for a in agents
    )
    return {"agents": agents, "trace": state["trace"] + [f"workload: {summary}"]}


def assign_agent(state: TicketState) -> TicketState:
    agents, cat = state["agents"], state["category"]
    skilled = [a for a in agents if cat in a["skills"]] or agents
    available = [a for a in skilled if a["open_count"] < db.MAX_OPEN] or skilled
    # lowest weighted load wins; ties -> fewer open tickets
    best = min(available, key=lambda a: (a["load_score"], a["open_count"]))
    reason = (
        f"{best['name']} handles '{cat}' and has the lightest load among skilled agents "
        f"({best['open_count']} open, {best['high_priority_count']} high-priority, load {best['load_score']})."
    )
    tid = db.save_ticket(state.get("customer", "anonymous"), state["text"], cat, state["priority"], best["id"])
    return {
        "assigned_agent": best["name"],
        "assignment_reason": reason,
        "ticket_id": tid,
        "trace": state["trace"] + [f"assign: ticket #{tid} -> {best['name']}"],
    }


def build_graph():
    g = StateGraph(TicketState)
    g.add_node("read_ticket", read_ticket)
    g.add_node("classify", classify)
    g.add_node("assign_priority", assign_priority)
    g.add_node("check_workload", check_workload)
    g.add_node("assign_agent", assign_agent)
    g.add_edge(START, "read_ticket")
    g.add_edge("read_ticket", "classify")
    g.add_edge("classify", "assign_priority")
    g.add_edge("assign_priority", "check_workload")
    g.add_edge("check_workload", "assign_agent")
    g.add_edge("assign_agent", END)
    return g.compile()
