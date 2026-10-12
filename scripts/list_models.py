"""Show which chat models YOUR token/account can actually call.
Run:  python -m scripts.list_models            (all)
      python -m scripts.list_models qwen       (filter by text)
"""
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise SystemExit("Set HUGGINGFACEHUB_API_TOKEN in .env first")

r = httpx.get(
    "https://router.huggingface.co/v1/models",
    headers={"Authorization": f"Bearer {token}"},
    timeout=30,
)
print("HTTP", r.status_code)
r.raise_for_status()

flt = sys.argv[1].lower() if len(sys.argv) > 1 else ""
ids = sorted(m["id"] for m in r.json().get("data", []))
shown = [i for i in ids if flt in i.lower()]
for i in shown:
    print(i)
print(f"\n{len(shown)} model(s) shown (of {len(ids)} total). Put one in .env as HF_MODEL=<id>")
