import numpy as np

rng = np.random.default_rng(7)
d = 8
vocab = ["doi", "dia", "chi", "khong", "huy", "don"]
E = {w: rng.normal(size=d) for w in vocab}  # embedding ngẫu nhiên, cố định theo seed
Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))


def self_attention(X):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(d)
    w = np.exp(scores - scores.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    return w @ V


def sinusoidal(n, d):
    pos = np.arange(n)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / 10000 ** (i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe


a = "doi dia chi khong huy don".split()  # đổi địa chỉ, không hủy đơn
b = "huy don khong doi dia chi".split()  # hủy đơn, không đổi địa chỉ
Xa = np.stack([E[w] for w in a])
Xb = np.stack([E[w] for w in b])

Ya, Yb = self_attention(Xa), self_attention(Xb)
print("token 'huy' in a:", Ya[a.index("huy")][:4].round(4))
print("token 'huy' in b:", Yb[b.index("huy")][:4].round(4))
print("max diff of mean-pooled vectors, no position:", float(np.abs(Ya.mean(0) - Yb.mean(0)).max()))

P = sinusoidal(len(a), d)
Za, Zb = self_attention(Xa + P), self_attention(Xb + P)  # cộng vị trí vào embedding
print("token 'huy' in a, with position:", Za[a.index("huy")][:4].round(4))
print("token 'huy' in b, with position:", Zb[b.index("huy")][:4].round(4))
print("max diff of mean-pooled vectors, with sinusoidal:", round(float(np.abs(Za.mean(0) - Zb.mean(0)).max()), 4))
