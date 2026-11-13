from pathlib import Path

import httpx

URL = "http://localhost:11434/api/chat"
POLICY = (Path(__file__).resolve().parents[1] / "data" / "support_policy.txt").read_text(encoding="utf-8")


def prefill(system, question):
    body = {"model": "qwen3:1.7b", "think": False, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
            "options": {"num_predict": 1, "temperature": 0, "seed": 7}}  # chỉ đo phần prefill
    d = httpx.post(URL, json=body, timeout=600).json()
    return d["prompt_eval_count"], d.get("prompt_eval_cached_count", 0), d["prompt_eval_duration"] / 1e9


cases = [
    ("1st customer", POLICY, "Phí giao 5 km tiêu chuẩn là bao nhiêu?"),
    ("2nd customer", POLICY, "Tôi muốn hủy đơn AT010570 được không?"),
    ("date on top", "Hôm nay: 05/11/2026.\n" + POLICY, "Tôi muốn hủy đơn AT010570 được không?"),
]
print(f"{'request':<14}{'in tok':>8}{'cached':>8}{'prefill s':>11}")
for name, system, question in cases:
    n, cached, secs = prefill(system, question)
    print(f"{name:<14}{n:>8}{cached:>8}{secs:>11.2f}")
