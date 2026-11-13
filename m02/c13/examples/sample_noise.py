import numpy as np

from _common import human_labels, load_golden

items = load_golden()
h = human_labels()
p1 = sum(h[(i["id"], "v1")] for i in items) / len(items)
p2 = sum(h[(i["id"], "v2")] for i in items) / len(items)
print(f"pass rate on 40 golden items (human labels): v1 {p1:.3f}, v2 {p2:.3f}")
rng = np.random.default_rng(13)
print(f"{'n':>4}  {'v2 measured (95% of runs)':>27}  {'P(v2 looks >= v1)':>18}")
for n in [5, 12, 40, 300]:
    a = rng.binomial(n, p1, 20_000) / n  # 20.000 lần thử, mỗi lần chấm n tin ngẫu nhiên
    b = rng.binomial(n, p2, 20_000) / n
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"{n:>4}  {lo:>13.2f} to {hi:<11.2f}  {np.mean(b >= a):>18.3f}")
