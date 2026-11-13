import _project  # noqa: F401

import anhti_chat.evals as ev
from anhti_chat.evals import check_gate, load_thresholds, run_eval

stores = []


class SpyStore(ev.MemoryStore):
    def __init__(self):
        super().__init__()
        stores.append(self)  # giữ lại kho của từng hội thoại để đọc tóm tắt


ev.MemoryStore = SpyStore
report = run_eval()
th = load_thresholds()
print("model:", report["model"], "| conversations:", report["metrics"]["n"], "| replayed from evals/cache")
for key, value in report["metrics"].items():
    limit = f">= {th['min'][key]}" if key in th.get("min", {}) else f"<= {th['max'][key]}" if key in th.get("max", {}) else ""
    print(f"  {key:<20} {value!s:<8} {limit}")
print("gate:", check_gate(report["metrics"], th) or "pass")
for r in report["rows"]:
    flags = [k[:-3] for k in ("intent_ok", "tools_ok", "facts_ok", "handoff_ok") if r.get(k) is False]
    if flags or r["replaced"] or r["violations"]:
        print(f"  {r['id']} intent={r['intent']:<14} miss={','.join(flags) or '-':<22} replaced={r['replaced'] or '-'}")
conv = next(iter(stores[-1].convs.values()))
print("E40 summary_upto:", conv["summary_upto"], "| summary:", conv["summary"])
print("E40 answer before guard:", next(r for r in run_eval(guard=False, only=["E40"])["rows"])["answer"])
