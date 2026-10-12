"""Measure classification accuracy and priority sanity on the labelled set.
Run:  python -m scripts.evaluate
Uses a throw-away DB so your real tickets are untouched.
"""
import json
import os
import tempfile
from collections import Counter

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "eval.db")

from app import db                      # noqa: E402  (env var must be set first)
from app.graph import build_graph       # noqa: E402

db.init_db()
graph = build_graph()
data = json.load(open("data/eval_tickets.json"))

cat_ok = pri_ok = 0
errors = []
for row in data:
    r = graph.invoke({"text": row["text"], "customer": "eval", "trace": []})
    c_ok = r["category"] == row["category"]
    p_ok = r["priority"] <= row["max_priority"]
    cat_ok += c_ok
    pri_ok += p_ok
    if not (c_ok and p_ok):
        errors.append((row["text"][:60], row["category"], r["category"], row["max_priority"], r["priority"]))

n = len(data)
print(f"Classification accuracy : {cat_ok}/{n} = {cat_ok / n:.0%}")
print(f"Priority within bound   : {pri_ok}/{n} = {pri_ok / n:.0%}")
if errors:
    print("\nMismatches (text | expected cat | got | max prio | got prio):")
    for e in errors:
        print(" -", e)

# fairness check: how evenly were new tickets spread?
load = Counter(a["name"] for a in db.get_agent_workloads() for _ in range(a["open_count"]))
print("\nOpen tickets per agent after run:", dict(load))
