import json
import tempfile
from pathlib import Path

import tiktoken

from _common import PROMPTS, golden, prices, wrap

enc = tiktoken.get_encoding("o200k_base")
p = prices()["api"]["models"]["gpt-6-luna"]
lines = []
for r in golden().itertuples():
    body = {"model": "gpt-6-luna", "instructions": PROMPTS["full"], "input": wrap(r.text),
            "reasoning": {"effort": "none"}, "max_output_tokens": 16}
    lines.append(json.dumps({"custom_id": r.message_id, "method": "POST", "url": "/v1/responses", "body": body},
                            ensure_ascii=False))
path = Path(tempfile.mkdtemp()) / "intent_batch.jsonl"
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(lines)} lines, {path.stat().st_size:,} bytes; first line starts with:")
print(lines[0][:118])

tok_in = sum(len(enc.encode(PROMPTS["full"] + wrap(r.text))) for r in golden().itertuples())
tok_out = 3 * len(lines)
for tier, q in [("standard", p), ("batch", p["batch"])]:
    usd = (tok_in * q["input"] + tok_out * q["output"]) / 1e6
    print(f"{tier:9} 63 messages ${usd:.6f}, scaled to 10,000 messages ${usd * 10_000 / 63:.4f}")
