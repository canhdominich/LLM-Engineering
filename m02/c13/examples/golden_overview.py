import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from checks import fact_check, policy_check
from golden import by_id, load_golden

items = load_golden()
print("items:", len(items))
print("intent:", dict(Counter(x["expected"]["intent"] for x in items)))
print("tags:", dict(Counter(t for x in items for t in x["tags"] if t != "single_turn")))
print("order is null:", sum(x["order"] is None for x in items), "| multi-turn:", sum(len(x["turns"]) > 1 for x in items))
g = by_id(items)["G37"]
print("G37 turns:", json.dumps(g["turns"], ensure_ascii=False))
print("G37 must:", g["expected"]["must"], "| forbid (last 4):", g["expected"]["forbid"][-4:])
bad = [x["id"] for x in items if fact_check(x, x["reference"]) or policy_check(x, x["reference"])]
print(f"references passing their own checks: {len(items) - len(bad)}/{len(items)}", bad or "")
