import math
import os

import httpx

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")


def next_token(prefix: str) -> list[tuple[str, float]]:
    body = {"model": MODEL, "prompt": prefix, "raw": True, "stream": False,  # raw: không bọc chat template
            "options": {"temperature": 0, "num_predict": 1, "seed": 7}, "logprobs": True, "top_logprobs": 4}
    r = httpx.post(f"{OLLAMA}/api/generate", json=body, timeout=120)
    r.raise_for_status()
    top = r.json()["logprobs"][0]["top_logprobs"]
    return [(t["token"], math.exp(t["logprob"])) for t in top]


text = "Đơn hàng của bạn đang được"
print(f"model {MODEL}, prefix {text!r}")
for step in range(6):
    candidates = next_token(text)
    chosen = candidates[0][0]  # greedy: lấy token có xác suất cao nhất
    shown = ", ".join(f"{tok!r} {p:.3f}" for tok, p in candidates)
    print(f"step {step + 1}: {shown}")
    text += chosen
print(f"result: {text!r}")

print("same last word 'giao', different context:")
for prefix in ["Shipper đang trên đường giao", "Tôi muốn đổi địa chỉ giao"]:
    shown = ", ".join(f"{tok!r} {p:.3f}" for tok, p in next_token(prefix))
    print(f"  {prefix!r}: {shown}")
