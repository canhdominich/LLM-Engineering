import numpy as np

from _bench import records

a = {r["message_id"]: r["pred"] == r["intent"] for r in records("tfidf")}
b = {r["message_id"]: r["pred"] == r["intent"] for r in records("qwen3-full")}
ids = sorted(a)
diff = np.array([b[i] - a[i] for i in ids], dtype=float)  # 1, 0 hoặc -1 cho từng tin
print(f"all 63: tfidf {np.mean(list(a.values())):.3f}, qwen3 {np.mean(list(b.values())):.3f}")

rng = np.random.default_rng(0)
for n in (6, 12, 30):
    picks = [rng.choice(len(ids), size=n, replace=False) for _ in range(2000)]  # 2.000 lần thử ngẫu nhiên
    flip = np.mean([diff[p].mean() >= 0 for p in picks])
    print(f"random {n:>2} messages: qwen3 ties or beats tfidf in {flip:.1%} of 2000 draws")

boot = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"qwen3 minus tfidf on 63: {diff.mean():+.3f}, 95% bootstrap interval [{lo:+.3f}, {hi:+.3f}]")
print(f"messages where only one is right: tfidf only {int((diff < 0).sum())}, qwen3 only {int((diff > 0).sum())}")
