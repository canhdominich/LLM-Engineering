import math
import re
from pathlib import Path

from _llm import MODEL, ask

POLICY = (Path(__file__).resolve().parents[1] / "data" / "fee_policy.txt").read_text(encoding="utf-8")
CASES = [(2.0, False, 1, False), (7.4, True, 3, False), (5.0, False, 8, True), (9.2, True, 6, True)]


def true_fee(km, express, kg, member):
    fee = 18000 + 4000 * max(0, math.ceil(km - 3)) + 15000 * express + 10000 * (kg > 5)
    return round(fee * 0.9) if member else fee


def question(km, express, kg, member):
    service = "express" if express else "standard"
    who = "a member" if member else "not a member"
    return f"Order: {km} km, {service}, {kg} kg, customer is {who}. What is the fee in VND?"


def read_number(text):
    # chỉ nhận "FINAL: n" ở cuối hoặc câu trả lời chỉ có một con số
    m = re.search(r"FINAL:\s*([\d.,]+)\s*$", text, re.I) or re.fullmatch(r"\s*([\d.,]+)\s*(VND)?\s*", text)
    return int(m.group(1).replace(",", "").replace(".", "")) if m else None


MODES = {
    "direct": (POLICY + "\nAnswer with the number only.", False, 30),
    "steps in prompt": (POLICY + "\nCompute step by step. End with a line: FINAL: <number>", False, 400),
    "thinking on": (POLICY + "\nAnswer with the number only.", True, 3000),
}
print(f"{'mode':16}{'correct':>8}{'out tok':>9}{'seconds':>9}")
answers, tokens = {}, {}
for mode, (instructions, think, limit) in MODES.items():
    ok = tok = sec = 0
    answers[mode], tokens[mode] = [], []
    for case in CASES:
        out = ask(instructions, question(*case), think=think, max_tokens=limit)
        got = read_number(out["text"])
        answers[mode].append(got)
        ok += got == true_fee(*case)
        tok += out["out_tok"]
        tokens[mode].append(out["out_tok"])
        sec += out["seconds"]
    print(f"{mode:16}{ok:>6}/4{tok / 4:>9.0f}{sec / 4:>9.1f}")
print("truth:", [true_fee(*c) for c in CASES])
for mode, got in answers.items():
    print(f"{mode}: {got}, out tokens {tokens[mode]}")
print(f"model {MODEL}")
