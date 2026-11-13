import math
from statistics import mean

import tiktoken

from _common import PROMPTS, golden, prices, run_candidate, wrap

enc = tiktoken.get_encoding("o200k_base")
cfg = prices()
host = cfg["self_host"]
tok_in = mean(len(enc.encode(PROMPTS["full"] + wrap(t))) for t in golden()["text"])
api = {m: (tok_in * p["input"] + 3 * p["output"]) / 1e6 for m, p in cfg["api"]["models"].items()}
fixed = host["server_usd_per_month"] + host["ops_hours_per_month"] * host["ops_usd_per_hour"]
secs = sorted(r["seconds"] for r in run_candidate("qwen3-full"))  # đo thật trên máy dùng chung của bài
print(f"self-host per server: ${fixed:.0f}/month = server + {host['ops_hours_per_month']} ops hours ({host['note']})")
print(f"API per call: gpt-6-luna ${api['gpt-6-luna']:.7f}, gpt-6.1-sol ${api['gpt-6.1-sol']:.7f}")

for label, s in [("shared CPU, p50", secs[len(secs) // 2]), ("quiet CPU, fastest call", secs[0])]:
    capacity = 30 * 24 * 3600 / s  # một máy chạy liên tục, mỗi lúc một yêu cầu
    print(f"{label}: {s:.2f} s per call, one server handles {capacity:,.0f} calls per month")
    print(f"  {'calls/month':>11}{'luna $':>9}{'sol $':>9}{'self-host $':>13}{'servers':>9}")
    for volume in (100_000, 300_000, 1_000_000, 3_000_000):
        n = math.ceil(volume / capacity)
        print(f"  {volume:>11,}{volume * api['gpt-6-luna']:>9.0f}{volume * api['gpt-6.1-sol']:>9.0f}{n * fixed:>13.0f}{n:>9}")
