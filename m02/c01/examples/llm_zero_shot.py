import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from keyword_rules import classify

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
LABELS = "order_status, eta, complaint_late, refund, change_address, other"
SHORT = f"Classify the customer message. Reply with exactly one label from: {LABELS}."
DEFINED = SHORT + """
Messages are Vietnamese, often typed without diacritics or with typos.
order_status: where the order is, picked up or delivered yet.
eta: when it will arrive, how long delivery takes.
complaint_late: the order is already late and the customer is unhappy.
refund: asks for money back or about a refund.
change_address: change address, receiver or drop-off place.
other: anything else."""

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", "ollama"), timeout=120)
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")


def ask(instructions: str, text: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = client.responses.create(
        model=MODEL, instructions=instructions, input=text, temperature=0,
        max_output_tokens=700, reasoning={"effort": "none"},  # tắt bước suy nghĩ của qwen3
    )
    return resp.output_text.strip().lower(), time.perf_counter() - t0


df = pd.read_csv(DATA)
sample = df[df["split"] == "test"].sort_values("message_id").groupby("intent").head(2)
score = {"rules": 0, "short": 0, "defined": 0}
seconds = []
print(f"{'message':30} {'true':14} {'rules':14} {'llm short':14} llm defined")
for _, r in sample.iterrows():
    preds = {"rules": classify(r["text"])}
    for name, instr in [("short", SHORT), ("defined", DEFINED)]:
        preds[name], sec = ask(instr, r["text"])
        seconds.append(sec)
    for k in score:
        score[k] += preds[k] == r["intent"]
    print(f"{r['text'][:28]!r:30} {r['intent']:14} {preds['rules']:14} {preds['short'][:14]:14} {preds['defined'][:14]}")
print(f"correct out of {len(sample)}: " + ", ".join(f"{k} {v}" for k, v in score.items()))
print(f"model {MODEL}, mean {sum(seconds) / len(seconds):.1f} s per LLM call")
