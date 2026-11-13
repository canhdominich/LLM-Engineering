import re

from golden import fold

MINUTES = re.compile(r"(\d+)\s*phut")


def allowed_minutes(item):
    o = item["order"] or {}
    ok = {60, 120, o.get("eta_minutes"), o.get("minutes_late")}
    for t in item["turns"]:
        ok |= {int(x) for x in MINUTES.findall(fold(t["content"]))}
    return {x for x in ok if x is not None}


def fact_check(item, reply):
    text = fold(reply)
    missing = [g[0] for g in item["expected"]["must"] if not any(fold(w) in text for w in g)]
    invented = [f"{x} phut" for x in map(int, MINUTES.findall(text)) if x not in allowed_minutes(item)]
    return missing + invented


def policy_check(item, reply):
    text = fold(reply)
    return [p for p in item["expected"]["forbid"] if fold(p) in text]


def score(item, pred):
    facts, policy = fact_check(item, pred["reply"]), policy_check(item, pred["reply"])
    intent_ok = pred["intent"] == item["expected"]["intent"]
    return {"id": item["id"], "intent_ok": intent_ok, "fact_ok": not facts, "policy_ok": not policy,
            "rules_ok": not facts and not policy, "problems": facts + policy}


def macro_f1(gold, pred):
    f1s = []
    for c in sorted(set(gold) | set(pred)):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def summarize(items, preds):
    rows = [score(it, preds[it["id"]]) for it in items]
    n = len(rows)
    gold = [it["expected"]["intent"] for it in items]
    pred = [preds[it["id"]]["intent"] for it in items]
    return {"n": n, "intent_acc": sum(r["intent_ok"] for r in rows) / n, "macro_f1": macro_f1(gold, pred),
            "fact_rate": sum(r["fact_ok"] for r in rows) / n, "policy_rate": sum(r["policy_ok"] for r in rows) / n,
            "rows": rows}
