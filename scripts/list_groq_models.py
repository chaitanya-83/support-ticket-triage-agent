import os
import httpx
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")
if not key:
    raise SystemExit("GROQ_API_KEY is missing in .env")
r = httpx.get("https://api.groq.com/openai/v1/models",
              headers={"Authorization": f"Bearer {key}"}, timeout=30)
print("HTTP", r.status_code)
if r.status_code != 200:
    print(r.text)
    raise SystemExit
for m in sorted(r.json()["data"], key=lambda x: x["id"]):
    print(m["id"])
