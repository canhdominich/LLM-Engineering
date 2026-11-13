from collections import Counter

from sklearn.metrics import f1_score

from _common import human_labels, load_golden, replies
from checks import summarize

items = load_golden()
human = human_labels()
gold = [it["expected"]["intent"] for it in items]
for v in ["v1", "v2"]:
    preds = replies(v, items)
    rep = summarize(items, preds)
    sk = f1_score(gold, [preds[i["id"]]["intent"] for i in items], average="macro")
    print(f"{v}: intent_acc {rep['intent_acc']:.3f}  macro_f1 {rep['macro_f1']:.3f} (sklearn {sk:.3f})  "
          f"fact_rate {rep['fact_rate']:.3f}  policy_rate {rep['policy_rate']:.3f}")
    agree = sum(r["rules_ok"] == human[(r["id"], v)] for r in rep["rows"])
    miss = [r["id"] for r in rep["rows"] if r["rules_ok"] and not human[(r["id"], v)]]
    print(f"    rules agree with human on {agree}/40; rules pass but human fails: {miss}")
    print("    top problems:", Counter(p for r in rep["rows"] for p in r["problems"]).most_common(4))
