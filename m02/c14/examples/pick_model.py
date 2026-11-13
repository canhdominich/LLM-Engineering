from statistics import mean

import tiktoken

from _bench import records, summary
from _common import PROMPTS, golden, prices, wrap

MIN_F1, MAX_P95 = 0.75, 5.0  # ngưỡng đặt trước khi nhìn kết quả
cfg = prices()
usd_per_hour = cfg["self_host"]["server_usd_per_month"] / (30 * 24)
enc = tiktoken.get_encoding("o200k_base")
tok_in = mean(len(enc.encode(PROMPTS["full"] + wrap(t))) for t in golden()["text"])

rows = []
for key in ["rules", "tfidf", "gemma3-full", "qwen3-short", "qwen3-full"]:
    s = summary(records(key))
    s.update(name=key, cost=1000 * s["p50"] / 3600 * usd_per_hour)  # giờ máy cho 1.000 lượt
    s["verdict"] = "quality" if s["f1"] < MIN_F1 else ("latency" if s["p95"] > MAX_P95 else "pass")
    rows.append(s)
for model, p in cfg["api"]["models"].items():
    cost = 1000 * (tok_in * p["input"] + 3 * p["output"]) / 1e6
    rows.append({"name": model, "f1": float("nan"), "p95": float("nan"), "cost": cost, "verdict": "not measured"})

print(f"threshold: macro F1 >= {MIN_F1}, p95 <= {MAX_P95} s; server {usd_per_hour:.4f} USD/hour (assumption)")
print(f"{'candidate':13}{'macroF1':>8}{'p95 s':>8}{'USD/1000':>10}  verdict")
for s in sorted(rows, key=lambda r: r["cost"]):
    print(f"{s['name']:13}{s['f1']:>8.3f}{s['p95']:>8.2f}{s['cost']:>10.5f}  {s['verdict']}")
ok = [s for s in rows if s["verdict"] == "pass"]
best = min(ok, key=lambda r: (r["cost"], r["p95"])) if ok else None
print("choice:", best["name"] if best else "none passes, change candidates or thresholds")
print(f"API prices checked on {cfg['api']['checked_on']}, {tok_in:.0f} input tokens per call (o200k_base)")
