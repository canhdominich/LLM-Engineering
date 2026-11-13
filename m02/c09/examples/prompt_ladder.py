from _data import parse_label, sample12
from _llm import MODEL, ask
from _prompts import PROMPTS, wrap

rows = sample12()
print(f"{'prompt':14}{'chars':>6}{'in tok':>7}{'exact':>6}{'parsed':>7}{'invalid':>8}{'other':>6}")
raw_v1, wrong = [], {}
for name, instructions in PROMPTS.items():
    exact = parsed = invalid = tokens = other = 0
    for _, r in rows.iterrows():
        text = wrap(r["text"]) if "<message>" in instructions else r["text"]
        out = ask(instructions, text)
        label = parse_label(out["text"])
        exact += out["text"] == r["intent"]
        parsed += label == r["intent"]
        invalid += label == "invalid"
        other += label == "other"
        tokens += out["in_tok"]
        if label != r["intent"]:
            wrong.setdefault(name, []).append(f"{r['message_id']} [{r['intent']}] got {label}")
        if name.startswith("v1") and len(raw_v1) < 3:
            raw_v1.append(out["text"].replace("\n", " ")[:70])
    print(f"{name:14}{len(instructions):>6}{tokens / len(rows):>7.0f}{exact:>6}{parsed:>7}{invalid:>8}{other:>6}")
print(f"messages: {len(rows)} (2 per intent, 2 are really other), model {MODEL}, reasoning off")
print("v1 raw replies:")
for x in raw_v1:
    print(" ", repr(x))
print("v3 wrong:", "; ".join(wrong["v3 + context"]))
print("v4 wrong:", "; ".join(wrong["v4 + rules"]))
