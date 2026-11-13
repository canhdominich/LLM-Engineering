import httpx

BASE = "http://localhost:11434/api"
PROMPT = "A customer asks: my parcel AT027075 is late, what should I do? Answer in two short sentences."
tags = {m["name"]: m for m in httpx.get(f"{BASE}/tags", timeout=60).json()["models"]}


def bits_per_weight(name):
    info = httpx.post(f"{BASE}/show", json={"model": name}, timeout=60).json()["model_info"]
    return tags[name]["size"] * 8 / info["general.parameter_count"]


def run(name):
    body = {"model": name, "prompt": PROMPT, "stream": False, "keep_alive": "30s",
            "options": {"num_predict": 48, "temperature": 0, "seed": 7}}
    httpx.post(f"{BASE}/generate", json={**body, "prompt": "hi", "options": {"num_predict": 1}}, timeout=300)  # nạp model trước
    d = httpx.post(f"{BASE}/generate", json=body, timeout=300).json()
    return d["prompt_eval_count"] / d["prompt_eval_duration"] * 1e9, d["eval_count"] / d["eval_duration"] * 1e9, d["response"]


print(f"{'model':<30}{'file MB':>8}{'bits/w':>7}{'prefill t/s':>12}{'decode t/s':>11}")
answers = {}
for name in ("smollm2:135m-instruct-fp16", "smollm2:135m-instruct-q8_0", "smollm2:135m-instruct-q4_K_M"):
    prefill, decode, answers[name] = run(name)
    print(f"{name:<30}{tags[name]['size'] / 1e6:>8.1f}{bits_per_weight(name):>7.2f}{prefill:>12.1f}{decode:>11.1f}")
print(f"{'qwen3:1.7b':<30}{tags['qwen3:1.7b']['size'] / 1e6:>8.1f}{bits_per_weight('qwen3:1.7b'):>7.2f}")
for name, text in answers.items():
    print(f"{name.split('-')[-1]:>5}: {' '.join(text.split())[:110]}")
