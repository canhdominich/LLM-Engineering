import hashlib
import json
import os
import time
from pathlib import Path

from openai import OpenAI

MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", "ollama"), timeout=900)


def _ask(instructions, text, think=False, max_tokens=60):
    kw = dict(model=MODEL, instructions=instructions, input=text, temperature=0, max_output_tokens=max_tokens)
    if not think:
        kw["reasoning"] = {"effort": "none"}  # tắt bước suy nghĩ của qwen3
    t0 = time.perf_counter()
    r = client.responses.create(**kw)
    thought = " ".join(s.text for o in r.output if o.type == "reasoning" for s in (o.summary or []))
    return {"text": r.output_text.strip(), "thought": thought.strip(),
            "in_tok": r.usage.input_tokens, "out_tok": r.usage.output_tokens,
            "seconds": round(time.perf_counter() - t0, 2)}


def ask(instructions, text, think=False, max_tokens=60):
    folder = os.environ.get("C09_CACHE")
    if not folder:
        return _ask(instructions, text, think, max_tokens)
    key = hashlib.sha256(json.dumps([MODEL, instructions, text, think, max_tokens]).encode()).hexdigest()[:24]
    path = Path(folder) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    out = _ask(instructions, text, think, max_tokens)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out
