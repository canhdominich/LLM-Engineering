from pathlib import Path

from transformers import AutoTokenizer

DATA = Path(__file__).resolve().parents[1] / "data"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def n_tokens(text):
    return len(tok.encode(text))


message = "don AT027075 tre 40 phut roi, hoa toc ma vay a, co duoc hoan phi khong"
order = "ORDER AT027075: express, Xom Vui, promised 60 min, picked up 10:05, now 10:45, not delivered."
faq = (DATA / "faq.txt").read_text(encoding="utf-8").splitlines()
history = [f"turn {i}: " + ("customer asks about a voucher code for next week" if i % 2 else "agent explains the voucher rules in detail and thanks the customer") for i in range(1, 25)]


def overlap(line):
    words = {"late", "express", "refund", "refunded", "surcharge"}  # từ khóa của ý định đang xử lý
    return len(words & set(line.lower().replace(".", "").split()))


# (phần, nội dung, ưu tiên): số nhỏ được giữ trước
parts = [("rules", (DATA / "support_rules.txt").read_text(encoding="utf-8"), 0), ("message", message, 0), ("order", order, 1)]
parts += [("faq", x, 2 - overlap(x) / 10 if overlap(x) else 5) for x in faq]
parts += [("history", h, 3 + (24 - i) / 100) for i, h in enumerate(history, start=1)]

BUDGET = 400
used, kept, dropped = 0, [], []
for name, text, prio in sorted(parts, key=lambda p: p[2]):
    cost = n_tokens(text)
    if used + cost <= BUDGET or prio == 0:
        used += cost
        kept.append((name, cost))
    else:
        dropped.append((name, cost))
print(f"everything: {sum(n_tokens(t) for _, t, _ in parts)} tokens, budget {BUDGET}, kept {used}")
print(f"{'part':10}{'kept':>6}{'of':>4}{'tokens kept':>13}{'tokens dropped':>16}")
for kind in ["rules", "message", "order", "faq", "history"]:
    k = [c for n, c in kept if n == kind]
    d = [c for n, c in dropped if n == kind]
    print(f"{kind:10}{len(k):>6}{len(k) + len(d):>4}{sum(k):>13}{sum(d):>16}")
print("faq kept:", [t[:38] for n, t, _ in sorted(parts, key=lambda p: p[2]) if n == "faq"][:2])
