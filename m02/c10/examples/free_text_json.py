import csv
import json
import os
from pathlib import Path

from openai import OpenAI

DATA = Path(__file__).resolve().parents[1] / "data"
client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", "ollama"), timeout=600)
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")

# Cách "xin" định dạng: chỉ dặn bằng lời trong prompt
INSTRUCTIONS = ("Extract the delivery request from the customer message. Reply in JSON with keys "
                "pickup_district, dropoff_district, items, cod_vnd, service.")
PICK = ["R02", "R06", "R09", "R12", "R17", "R25"]

rows = {r["request_id"]: r for r in csv.DictReader(open(DATA / "delivery_requests.csv", encoding="utf-8"))}
replies = {}
for rid in PICK:
    resp = client.responses.create(model=MODEL, instructions=INSTRUCTIONS, input=rows[rid]["text"],
                                   temperature=0, max_output_tokens=300, reasoning={"effort": "none"})
    replies[rid] = resp.output_text
    try:
        json.loads(resp.output_text)
        status = "json.loads ok"
    except json.JSONDecodeError as e:
        status = f"JSONDecodeError: {e.msg}"
    print(f"{rid} {status}")
    print("   ", repr(resp.output_text[:90]))

(DATA / "free_text_replies.json").write_text(json.dumps(replies, ensure_ascii=False, indent=1), encoding="utf-8")
print("model:", MODEL)
