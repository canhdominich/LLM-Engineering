import random
from pathlib import Path

from transformers import AutoTokenizer

from _llm import MODEL, ask

RULES = (Path(__file__).resolve().parents[1] / "data" / "support_rules.txt").read_text(encoding="utf-8")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
rnd = random.Random(7)
DISTRICTS = ["Dau Pho", "Cau Go", "Xom Vui", "Ben Song", "Bai Bien", "Cho Moi", "Doi Thong", "Khu Cong Nghiep"]
STATUS = ["picked up", "in transit", "delivered", "cancelled"]
orders = [f"AT{100000 + i} | {rnd.choice(STATUS)} | {rnd.choice(DISTRICTS)} | updated {rnd.randint(7, 21)}:{rnd.randint(0, 59):02d}"
          for i in range(300)]
orders[7] = "AT100007 | in transit | Xom Vui | updated 10:45"
question = "Khach hoi: don AT100007 dang o dau?"

for name, rows in [("all 300 orders", orders), ("only the asked order", [r for r in orders if r.startswith("AT100007")])]:
    text = "ORDER TABLE:\n" + "\n".join(rows) + "\n\n" + question
    counted = len(tok.encode(RULES + text))
    out = ask(RULES, text, max_tokens=80)
    print(f"{name}: counted {counted} tokens, server read {out['in_tok']}, {out['seconds']:.0f} s")
    print("  reply:", out["text"].replace("\n", " ")[:150])
print("truth: AT100007 | in transit | Xom Vui | updated 10:45")
print(f"model {MODEL}")
