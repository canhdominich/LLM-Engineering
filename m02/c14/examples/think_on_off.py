import numpy as np

from _common import run_candidate, think_subset

rows = think_subset()  # 12 tin, 2 mỗi ý định, khác template
print(f"{'mode':13}{'correct':>8}{'invalid':>8}{'out tok':>8}{'p50 s':>8}{'p95 s':>8}{'max s':>8}")
for name, mode in [("qwen3-full", "thinking off"), ("qwen3-full-think", "thinking on")]:
    recs = run_candidate(name, rows)
    secs = [r["seconds"] for r in recs]
    ok = sum(r["pred"] == r["intent"] for r in recs)
    bad = sum(r["pred"] == "invalid" for r in recs)
    print(f"{mode:13}{ok:>6}/12{bad:>8}{np.mean([r['out_tok'] for r in recs]):>8.0f}"
          f"{np.percentile(secs, 50):>8.1f}{np.percentile(secs, 95):>8.1f}{max(secs):>8.1f}")
    if name.endswith("think"):
        cut = [r["message_id"] for r in recs if r["out_tok"] >= 1000]
        print(f"hit the 1000-token limit: {cut or 'none'}")
        diff = [(r["message_id"], r["intent"], r["pred"]) for r in recs if r["pred"] != r["intent"]]
        print("wrong with thinking on:", diff)
