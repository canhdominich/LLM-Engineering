from _data import LABELS, parse_label, sample12
from _llm import MODEL, ask
from _prompts import wrap
from prompt_registry import load

rows = sample12()
preds = {}
for name in ["classify_v4", "classify_v5"]:
    tpl, version = load(name)
    instructions = tpl.substitute(labels=", ".join(LABELS)).strip()
    preds[name] = [parse_label(ask(instructions, wrap(t))["text"]) for t in rows["text"]]
    acc = sum(p == g for p, g in zip(preds[name], rows["intent"]))
    print(f"{name} ({version}): {acc}/{len(rows)} correct")

fixed, broken = [], []
for (_, r), a, b in zip(rows.iterrows(), preds["classify_v4"], preds["classify_v5"]):
    if a != r["intent"] and b == r["intent"]:
        fixed.append(r["message_id"])
    if a == r["intent"] and b != r["intent"]:
        broken.append(r["message_id"])
        print(f"  broken {r['message_id']} [{r['intent']}] now {b}: {r['text'][:44]}")
print("fixed:", fixed or "none", "| broken:", broken or "none")
verdict = "ship" if not broken and len(fixed) > 0 else "block"
print("gate (no broken case, at least one fix):", verdict)
print(f"model {MODEL}")
