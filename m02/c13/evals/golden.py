import json
import re
import unicodedata
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden.jsonl"
INTENTS = ["order_status", "eta", "complaint_late", "refund", "change_address", "other"]


def load_golden(path=GOLDEN):
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    ids = [x["id"] for x in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert all(x["expected"]["intent"] in INTENTS for x in items), "unknown intent"
    return items


def fold(text):
    # bỏ dấu, chữ thường, để "25 phút" và "25 phut" so khớp như nhau
    s = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)


def by_id(items):
    return {x["id"]: x for x in items}
