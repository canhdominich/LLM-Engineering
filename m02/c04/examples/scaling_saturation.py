import numpy as np

rng = np.random.default_rng(0)
n, trials = 8, 2000


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


print(" d_k  scaled  std(score)  mean max weight  rows > 0.99  slope w(1-w)")
for d_k in (4, 64, 512):
    q = rng.normal(size=(trials, n, d_k))
    k = rng.normal(size=(trials, n, d_k))
    raw = q @ k.transpose(0, 2, 1)  # (trials, n, n)
    for scaled in (False, True):
        s = raw / np.sqrt(d_k) if scaled else raw
        w_max = softmax(s).max(axis=-1)
        slope = (w_max * (1 - w_max)).mean()  # đạo hàm của trọng số lớn nhất theo điểm của nó
        print(f"{d_k:4d}  {str(scaled):6s}  {s.std():10.2f}  {w_max.mean():15.3f}  "
              f"{(w_max > 0.99).mean():11.3f}  {slope:12.3f}")
