from _data import parse_label, sample12
from _llm import MODEL, ask
from _prompts import BASE, wrap

rows = sample12().groupby("intent", sort=False).tail(1)  # 6 tin, 1 mỗi ý định
print(f"{'mode':14}{'correct':>8}{'out tok':>9}{'mean s':>8}{'max s':>7}")
thoughts = []
for mode, think, limit in [("thinking off", False, 60), ("thinking on", True, 1500)]:
    ok, toks, secs = 0, [], []
    for _, r in rows.iterrows():
        out = ask(BASE, wrap(r["text"]), think=think, max_tokens=limit)
        ok += parse_label(out["text"]) == r["intent"]
        toks.append(out["out_tok"])
        secs.append(out["seconds"])
        if think and r["message_id"] == "M0130":
            thoughts.append(out["thought"])
    print(f"{mode:14}{ok:>6}/6{sum(toks) / len(rows):>9.0f}{sum(secs) / len(rows):>8.1f}{max(secs):>7.1f}")
print(f"model {MODEL}")
if thoughts:
    t = thoughts[0].replace("\n", " ")
    print(f"thought for M0130 ({len(t)} chars), start: {t[:230]}")
