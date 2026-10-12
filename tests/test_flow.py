import os
import tempfile

os.environ["USE_LLM"] = "false"
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

from app import db                   # noqa: E402
from app.graph import build_graph    # noqa: E402

db.init_db()
graph = build_graph()


def run(text):
    return graph.invoke({"text": text, "customer": "t", "trace": []})


def test_urgent_refund_goes_to_lighter_agent():
    r = run("I need a refund immediately, this is urgent")
    assert r["category"] == "refund"
    assert r["priority"] <= 2
    assert r["assigned_agent"] == "Kumar"      # Deepak has high-priority tickets


def test_trace_has_all_steps():
    r = run("My package has not arrived")
    steps = [t.split(":")[0] for t in r["trace"]]
    assert steps == ["read_ticket", "classify", "priority", "workload", "assign"]


def test_unknown_text_falls_back_to_other():
    assert run("hello there, just saying hi")["category"] == "other"


def test_ticket_is_persisted():
    r = run("The item arrived broken")
    assert any(t["id"] == r["ticket_id"] for t in db.list_tickets())
