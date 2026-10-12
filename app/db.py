import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.getenv("DB_PATH", "tickets.db")
PRIORITY_WEIGHT = {1: 5, 2: 3, 3: 2, 4: 1}  # higher priority = heavier load
MAX_OPEN = 8                                  # agents at/over this are skipped if others are free


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            skills TEXT NOT NULL            -- comma-separated categories
        );
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer TEXT,
            text TEXT NOT NULL,
            category TEXT,
            priority INTEGER,
            assigned_to INTEGER REFERENCES agents(id),
            status TEXT DEFAULT 'open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        if c.execute("SELECT COUNT(*) FROM agents").fetchone()[0] == 0:
            _seed(c)


def _seed(c):
    agents = [
        ("Kumar", "refund,billing,damaged"),
        ("Deepak", "refund,billing,technical"),
        ("Anita", "delivery,damaged,account"),
        ("Priya", "technical,account,other"),
    ]
    c.executemany("INSERT INTO agents(name, skills) VALUES (?,?)", agents)
    ids = {r["name"]: r["id"] for r in c.execute("SELECT id, name FROM agents")}
    # Matches the resume example: Kumar 3 open (no high priority), Deepak 5 open (incl. high priority)
    seed = {
        "Kumar":  [(3, "refund"), (3, "billing"), (4, "damaged")],
        "Deepak": [(1, "billing"), (2, "refund"), (3, "technical"), (3, "billing"), (4, "refund")],
        "Anita":  [(2, "delivery"), (3, "account"), (3, "damaged"), (4, "delivery")],
        "Priya":  [(3, "technical"), (4, "other")],
    }
    for name, items in seed.items():
        for prio, cat in items:
            c.execute(
                "INSERT INTO tickets(customer,text,category,priority,assigned_to) VALUES (?,?,?,?,?)",
                ("seed", f"seed {cat} ticket", cat, prio, ids[name]),
            )


def get_agent_workloads():
    """Each agent with open ticket count, high-priority count (P1/P2) and a weighted load score."""
    with conn() as c:
        agents = [dict(r) for r in c.execute("SELECT * FROM agents")]
        for a in agents:
            rows = c.execute(
                "SELECT priority FROM tickets WHERE assigned_to=? AND status='open'", (a["id"],)
            ).fetchall()
            prios = [r["priority"] for r in rows]
            a["skills"] = a["skills"].split(",")
            a["open_count"] = len(prios)
            a["high_priority_count"] = sum(1 for p in prios if p <= 2)
            a["load_score"] = sum(PRIORITY_WEIGHT.get(p, 1) for p in prios)
    return agents


def save_ticket(customer, text, category, priority, agent_id) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO tickets(customer,text,category,priority,assigned_to) VALUES (?,?,?,?,?)",
            (customer, text, category, priority, agent_id),
        )
        return cur.lastrowid


def list_tickets(limit=50):
    with conn() as c:
        rows = c.execute(
            """SELECT t.id, t.customer, t.text, t.category, t.priority, t.status, a.name AS agent
               FROM tickets t LEFT JOIN agents a ON a.id=t.assigned_to
               ORDER BY t.id DESC LIMIT ?""", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def reset_db():
    """Wipe tickets/agents and re-seed the demo data (handy before a live demo)."""
    with conn() as c:
        c.execute("DELETE FROM tickets")
        c.execute("DELETE FROM agents")
        c.execute("DELETE FROM sqlite_sequence")
        _seed(c)
