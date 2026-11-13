import json
from typing import Literal

from pydantic import BaseModel, Field

from bot import POLICY, cached_call

HOLISTIC = """You grade replies of the support assistant of Giao Hang Anh Ti, a delivery company.
Rate the REPLY from 1 (very bad) to 5 (excellent) for overall quality. Give a one-sentence reason first.
""" + POLICY

CHECKLIST = """You grade replies of the support assistant of Giao Hang Anh Ti, a delivery company.
Check the REPLY against three criteria and answer each with true or false.
facts_correct: every order id, time, number of minutes, status, name and price in REPLY matches ORDER or POLICY. Anything not in ORDER or POLICY is invented, so false.
answers_question: REPLY addresses the last customer message. If ORDER is null and the customer asks about one order, REPLY must ask for the order id.
follows_policy: REPLY promises no refund, compensation or voucher, and does not claim an address was already changed.
A REFERENCE reply that passes all criteria is given; REPLY may use different words and still pass.
Write the problem in a few words first (or "none"), then the three answers. If unsure, answer false.
""" + POLICY

PAIRWISE = """You compare two replies, A and B, of the support assistant of Giao Hang Anh Ti, a delivery company.
Pick the reply that is more correct and helpful for the last customer message, using ORDER and POLICY.
""" + POLICY


class Holistic(BaseModel):
    reason: str
    score: int = Field(ge=1, le=5)


class Checklist(BaseModel):
    problem: str
    facts_correct: bool
    answers_question: bool
    follows_policy: bool


class Pairwise(BaseModel):
    reason: str
    winner: Literal["A", "B"]


def context(item):
    lines = [f"ORDER: {json.dumps(item['order'], ensure_ascii=False)}", "", "CONVERSATION:"]
    return "\n".join(lines + [f"{t['role']}: {t['content']}" for t in item["turns"]])


def judge_holistic(item, reply):
    text = context(item) + f"\n\nREPLY: {reply}"
    out = cached_call("judge_holistic", HOLISTIC, text, Holistic, 200)
    p = out["parsed"]
    return {"pass": bool(p and p["score"] >= 4), "detail": p, "in_tok": out["in_tok"], "out_tok": out["out_tok"],
            "seconds": out["seconds"]}


def judge_checklist(item, reply):
    text = context(item) + f"\n\nREFERENCE: {item['reference']}\n\nREPLY: {reply}"
    out = cached_call("judge_checklist", CHECKLIST, text, Checklist, 200)
    p = out["parsed"]
    ok = bool(p and p["facts_correct"] and p["answers_question"] and p["follows_policy"])
    return {"pass": ok, "detail": p, "in_tok": out["in_tok"], "out_tok": out["out_tok"], "seconds": out["seconds"]}


def judge_pair(item, reply_a, reply_b):
    text = context(item) + f"\n\nREPLY A: {reply_a}\n\nREPLY B: {reply_b}"
    out = cached_call("judge_pair", PAIRWISE, text, Pairwise, 200)
    return (out["parsed"] or {}).get("winner")
