from pathlib import Path

import httpx

URL = "http://localhost:11434/api/generate"
POLICY = (Path(__file__).resolve().parents[1] / "data" / "support_policy.txt").read_text(encoding="utf-8")
SHORT = "Khách hỏi: đơn AT027075 đang ở đâu?"
LONG = "Khách hỏi: phí giao 7 km bằng hỏa tốc là bao nhiêu?\n\n" + POLICY


def run(prompt):
    body = {"model": "qwen3:1.7b", "prompt": prompt, "think": False, "stream": False,
            "options": {"num_predict": 40, "temperature": 0, "seed": 7}}
    d = httpx.post(URL, json=body, timeout=600).json()
    ns = 1e9  # Ollama trả thời gian theo nano giây
    return {"prompt_tokens": d["prompt_eval_count"], "cached": d.get("prompt_eval_cached_count", 0),
            "prefill_s": d["prompt_eval_duration"] / ns, "out_tokens": d["eval_count"],
            "decode_s": d["eval_duration"] / ns, "load_s": d["load_duration"] / ns}


print(f"{'prompt':<14}{'in tok':>7}{'cached':>7}{'prefill s':>10}{'prefill tok/s':>14}"
      f"{'out tok':>8}{'decode s':>9}{'decode tok/s':>13}")
for name, prompt in [("short", SHORT), ("long", LONG)]:
    r = run(prompt)
    print(f"{name:<14}{r['prompt_tokens']:>7}{r['cached']:>7}{r['prefill_s']:>10.2f}"
          f"{r['prompt_tokens'] / r['prefill_s']:>14.1f}{r['out_tokens']:>8}{r['decode_s']:>9.2f}"
          f"{r['out_tokens'] / r['decode_s']:>13.1f}")
