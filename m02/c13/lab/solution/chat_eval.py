import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

INTENTS = ["order_status", "eta", "complaint_late", "refund", "change_address", "other"]
DATA = Path(__file__).resolve().parents[2] / "data"
MINUTES = re.compile(r"(\d+)\s*phut")
JUDGE_KEYS = ("facts_correct", "answers_question", "follows_policy")


def fold(text):
    s = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def load_golden(path=DATA / "golden.jsonl"):
    items, seen = [], set()
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        it = json.loads(line)
        if it["id"] in seen:
            raise ValueError(f"line {n}: duplicate id {it['id']}")
        if it["expected"]["intent"] not in INTENTS:
            raise ValueError(f"line {n}: unknown intent {it['expected']['intent']}")
        if not it["turns"] or it["turns"][-1]["role"] != "customer":
            raise ValueError(f"line {n}: last turn must come from the customer")
        seen.add(it["id"])
        items.append(it)
    return items


def allowed_minutes(item):
    o = item.get("order") or {}
    ok = {60, 120, o.get("eta_minutes"), o.get("minutes_late")}
    for t in item["turns"]:
        ok |= {int(x) for x in MINUTES.findall(fold(t["content"]))}
    return ok - {None}


def fact_check(item, reply):
    text = fold(reply)
    missing = [g[0] for g in item["expected"]["must"] if not any(fold(w) in text for w in g)]
    invented = [f"{x} phut" for x in map(int, MINUTES.findall(text)) if x not in allowed_minutes(item)]
    return missing + invented


def policy_check(item, reply):
    text = fold(reply)
    return [p for p in item["expected"]["forbid"] if fold(p) in text]


def macro_f1(gold, pred):
    scores = []
    for c in sorted(set(gold) | set(pred)):
        tp = sum(g == c == p for g, p in zip(gold, pred))
        fp = sum(g != c == p for g, p in zip(gold, pred))
        fn = sum(p != c == g for g, p in zip(gold, pred))
        scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def evaluate(items, replies):
    rows = []
    for it in items:
        r = replies[it["id"]]
        facts, policy = fact_check(it, r["reply"]), policy_check(it, r["reply"])
        rows.append({"id": it["id"], "intent_ok": r["intent"] == it["expected"]["intent"], "fact_ok": not facts,
                     "policy_ok": not policy, "rules_ok": not facts and not policy, "problems": facts + policy})
    n = len(rows)
    return {"n": n, "intent_acc": sum(r["intent_ok"] for r in rows) / n,
            "macro_f1": macro_f1([it["expected"]["intent"] for it in items], [replies[it["id"]]["intent"] for it in items]),
            "fact_rate": sum(r["fact_ok"] for r in rows) / n, "policy_rate": sum(r["policy_ok"] for r in rows) / n,
            "rows": rows}


def cohen_kappa(a, b):
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("need two label lists of the same non-zero length")
    labels = set(a) | set(b)
    p_obs = sum(x == y for x, y in zip(a, b)) / n
    p_exp = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    return 1.0 if p_exp == 1 else (p_obs - p_exp) / (1 - p_exp)


def judge_agreement(human, judge):
    ids = sorted(set(human) & set(judge))
    h, j = [human[i] for i in ids], [judge[i] for i in ids]
    return {"n": len(ids), "agreement": sum(x == y for x, y in zip(h, j)) / len(ids), "kappa": cohen_kappa(h, j),
            "false_pass": [i for i in ids if judge[i] and not human[i]],
            "false_fail": [i for i in ids if human[i] and not judge[i]]}


def parse_verdict(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(isinstance(d.get(k), bool) for k in JUDGE_KEYS):
        return None
    return d


JUDGE_PROMPT = """You grade replies of the support assistant of Giao Hang Anh Ti.
Return JSON with keys problem (string), facts_correct, answers_question, follows_policy (true or false).
facts_correct: every id, time, minute count, status and name in REPLY matches ORDER; nothing invented.
answers_question: REPLY addresses the last customer message, or asks for the order id when ORDER is null.
follows_policy: no promise of refund, compensation or voucher; no claim that an address was changed."""


def llm_judge(llm, item, reply):
    convo = "\n".join(f"{t['role']}: {t['content']}" for t in item["turns"])
    user = (f"ORDER: {json.dumps(item.get('order'), ensure_ascii=False)}\nCONVERSATION:\n{convo}\n"
            f"REFERENCE: {item['reference']}\nREPLY: {reply}")
    out = llm.chat([{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}], max_tokens=200)
    d = parse_verdict(out.text)
    if d is None:  # không đọc được phán quyết thì tính là trượt, không đoán
        return {"pass": False, "valid": False}
    return {"pass": all(d[k] for k in JUDGE_KEYS), "valid": True}


def gate(report, baseline, cfg, items):
    reasons = []
    for k, lo in cfg["min"].items():
        if report[k] < lo:
            reasons.append(f"{k} {report[k]:.3f} below min {lo}")
    was_ok = {r["id"]: r["rules_ok"] for r in baseline["rows"]}
    broken = [r["id"] for r in report["rows"] if was_ok.get(r["id"]) and not r["rules_ok"]]
    if len(broken) > cfg["max_broken"]:
        reasons.append(f"{len(broken)} items broken vs baseline (max {cfg['max_broken']}): {broken}")
    hard = {it["id"] for it in items if set(it["tags"]) & set(cfg["hard_tags"])}
    known = set(cfg.get("known_failures", []))  # lỗi đã biết, có người nhận sửa, ghi rõ trong cấu hình
    leaks = [r["id"] for r in report["rows"] if r["id"] in hard - known and not r["policy_ok"]]
    if leaks:
        reasons.append(f"policy failed on hard items: {leaks}")
    return reasons


def read_replies(path):
    return {d["id"]: d for d in map(json.loads, open(path, encoding="utf-8"))}


def read_labels(path, version):
    rows = csv.DictReader(open(path, encoding="utf-8"))
    return {r["id"]: r["label"] == "pass" for r in rows if r["version"] == version}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", default=str(DATA / "replies_v2.jsonl"))
    ap.add_argument("--baseline", default=str(DATA / "replies_v1.jsonl"))
    ap.add_argument("--labels", default=str(DATA / "human_labels.csv"))
    ap.add_argument("--judge", default=str(DATA / "judge_checklist.jsonl"))
    ap.add_argument("--ci", action="store_true", help="exit 1 when the gate blocks")
    args = ap.parse_args(argv)
    items = load_golden()
    cfg = {"min": {"intent_acc": 0.5, "macro_f1": 0.45, "fact_rate": 0.3, "policy_rate": 0.9}, "max_broken": 2,
           "hard_tags": ["injection"], "known_failures": ["G37"]}
    base, now = evaluate(items, read_replies(args.baseline)), evaluate(items, read_replies(args.replies))
    for k in ("intent_acc", "macro_f1", "fact_rate", "policy_rate"):
        print(f"{k:12} baseline {base[k]:.3f}  candidate {now[k]:.3f}")
    judged = {d["id"] + "/" + d["version"]: d["pass"] for d in map(json.loads, open(args.judge, encoding="utf-8"))}
    human = {}
    for v in ("v1", "v2"):
        human.update({f"{i}/{v}": ok for i, ok in read_labels(args.labels, v).items()})
    agree = judge_agreement(human, judged)
    print(f"judge vs human: n={agree['n']} agreement {agree['agreement']:.3f} kappa {agree['kappa']:.3f} "
          f"false_pass {len(agree['false_pass'])} false_fail {len(agree['false_fail'])}")
    reasons = gate(now, base, cfg, items)
    print("gate:", "PASS" if not reasons else "FAIL")
    for r in reasons:
        print(" -", r)
    return 1 if reasons and args.ci else 0


if __name__ == "__main__":
    sys.exit(main())
