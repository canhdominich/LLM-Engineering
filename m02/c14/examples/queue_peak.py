import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI

from _common import BASE_URL, CACHE, PROMPTS, golden, wrap

client = OpenAI(base_url=BASE_URL, api_key="ollama", timeout=900, max_retries=0)
LOG = CACHE / "queue-peak.json"


def one(text):
    t0 = time.perf_counter()
    client.responses.create(model="qwen3:1.7b", instructions=PROMPTS["full"], input=wrap(text),
                            reasoning={"effort": "none"}, max_output_tokens=12, temperature=0)
    return round(time.perf_counter() - t0, 2)


def experiment():
    texts = list(golden()["text"][10:26])
    out = {"serial": [one(t) for t in texts[:4]]}
    for n, part in [(4, texts[4:8]), (8, texts[8:16])]:
        with ThreadPoolExecutor(max_workers=n) as pool:  # n khách gửi cùng lúc
            out[f"burst {n}"] = list(pool.map(one, part))
    return out


log = json.loads(LOG.read_text()) if LOG.exists() else experiment()
LOG.write_text(json.dumps(log))
for name, secs in log.items():
    print(f"{name:8} each request: {secs}  p50 {np.percentile(secs, 50):.2f} s  max {max(secs):.2f} s")
base = np.percentile(log["serial"], 50)
for name in ("burst 4", "burst 8"):
    print(f"{name}: slowest / median alone = {max(log[name]) / base:.1f}x")
