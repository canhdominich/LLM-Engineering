import csv
import os
from pathlib import Path

from openai import OpenAI

from _delivery import DeliveryRequest

DATA = Path(__file__).resolve().parents[1] / "data"
client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", "ollama"), timeout=600)
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
INSTRUCTIONS = "Extract the delivery request from the customer message."  # không cần dặn định dạng nữa
PICK = ["R02", "R06", "R09", "R12", "R17", "R25"]
FIELDS = list(DeliveryRequest.model_fields)

rows = {r["request_id"]: r for r in csv.DictReader(open(DATA / "delivery_requests.csv", encoding="utf-8"))}
correct = 0
for rid in PICK:
    resp = client.responses.parse(model=MODEL, instructions=INSTRUCTIONS, input=rows[rid]["text"],
                                  text_format=DeliveryRequest, temperature=0, max_output_tokens=300,
                                  reasoning={"effort": "none"})
    req = resp.output_parsed  # đã là DeliveryRequest, không cần json.loads
    wrong = [f for f in FIELDS if str(getattr(req, f)) != rows[rid][f]]
    correct += len(FIELDS) - len(wrong)
    print(f"{rid} {req.pickup_district} > {req.dropoff_district}, {req.items} items, cod {req.cod_vnd}, {req.service}")
    print(f"    wrong: {wrong or 'none'}")
print(f"valid objects: {len(PICK)}/{len(PICK)}, correct fields: {correct}/{len(PICK) * len(FIELDS)}")
print("model:", MODEL)
