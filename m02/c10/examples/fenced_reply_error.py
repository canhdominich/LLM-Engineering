import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
replies = json.loads((DATA / "free_text_replies.json").read_text(encoding="utf-8"))

reply = replies["R02"]  # câu trả lời thật của model, bắt đầu bằng khung code markdown
order = json.loads(reply)
print(order["cod_vnd"])
