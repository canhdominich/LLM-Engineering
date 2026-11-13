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
    # TODO 1: đọc từng dòng JSON, bỏ dòng trống; báo ValueError khi id trùng, intent lạ
    # hoặc lượt cuối không phải của customer (thông báo có chữ "duplicate", "unknown intent", "last turn")
    raise NotImplementedError

def allowed_minutes(item):
    o = item.get("order") or {}
    ok = {60, 120, o.get("eta_minutes"), o.get("minutes_late")}
    for t in item["turns"]:
        ok |= {int(x) for x in MINUTES.findall(fold(t["content"]))}
    return ok - {None}


def fact_check(item, reply):
    # TODO 2: trả list vấn đề: nhóm từ bắt buộc bị thiếu (lấy phần tử đầu của nhóm)
    # và mọi "N phut" trong reply mà N không thuộc allowed_minutes(item); so khớp sau fold()
    raise NotImplementedError

def policy_check(item, reply):
    # TODO 3: trả các cụm trong expected.forbid xuất hiện trong reply (sau fold())
    raise NotImplementedError

def macro_f1(gold, pred):
    # TODO 4: F1 của từng nhãn có mặt trong gold hoặc pred, rồi lấy trung bình cộng
    raise NotImplementedError

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
    # TODO 5: kappa = (p_obs - p_exp) / (1 - p_exp); p_exp tính từ tần suất nhãn của từng bên
    # báo ValueError khi hai list rỗng hoặc khác độ dài
    raise NotImplementedError

def judge_agreement(human, judge):
    # TODO 6: trên các id có ở cả hai dict, trả n, agreement, kappa,
    # false_pass (judge đạt, người trượt) và false_fail (judge trượt, người đạt)
    raise NotImplementedError

def parse_verdict(text):
    # TODO 7: tìm khối {...} đầu tiên trong text, json.loads; trả None khi không đọc được
    # hoặc thiếu một trong JUDGE_KEYS kiểu bool
    raise NotImplementedError

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
    # TODO 8: trả list lý do chặn: metric dưới cfg["min"], số mục hỏng so với baseline
    # vượt cfg["max_broken"] (chữ "broken"), mục có tag trong cfg["hard_tags"] trượt policy (chữ "hard items"),
    # trừ các id trong cfg.get("known_failures", [])
    raise NotImplementedError

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
