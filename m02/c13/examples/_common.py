import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from bot import MODEL, answer
from golden import by_id, load_golden

LABELS = ROOT / "data" / "human_labels.csv"


def human_labels():
    # {(id, version): True nếu người chấm cho đạt}
    return {(r["id"], r["version"]): r["label"] == "pass" for r in csv.DictReader(open(LABELS, encoding="utf-8"))}


def replies(version, items):
    return {it["id"]: answer(it, version) for it in items}
