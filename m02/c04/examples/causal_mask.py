import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(5)
TOKENS = ["toi", "muon", "huy", "don", "AT027075"]
n, d = len(TOKENS), 8
Q, K, V = (rng.normal(size=(n, d)) for _ in range(3))


def attention(Q, K, V, causal):
    s = Q @ K.T / np.sqrt(d)
    if causal:
        future = np.triu(np.ones((n, n), dtype=bool), k=1)  # True ở các ô j > i
        s = np.where(future, -np.inf, s)                     # exp(-inf) = 0
    A = np.exp(s - s.max(axis=-1, keepdims=True))
    A = A / A.sum(axis=-1, keepdims=True)
    return A @ V, A


out, A = attention(Q, K, V, causal=True)
print("causal weights (row = token reading, col = token read):")
for tok, row in zip(TOKENS, A):
    print(f"  {tok:9s}", " ".join(f"{w:.2f}" for w in row))

_, A_full = attention(Q, K, V, causal=False)
print("row 'don' without mask:", " ".join(f"{w:.2f}" for w in A_full[3]))

V2 = V.copy()
V2[-1] += 10.0          # đổi hẳn token cuối "AT027075"
K2 = K.copy()
K2[-1] += 10.0
for causal in (False, True):
    a, _ = attention(Q, K, V, causal)
    b, _ = attention(Q, K2, V2, causal)
    print(f"causal={causal!s:5s} change at positions 0-3 when last token changes:",
          np.abs(a - b)[:-1].max(axis=1).round(3))

ref = F.scaled_dot_product_attention(*(torch.from_numpy(M)[None] for M in (Q, K, V)), is_causal=True)
print("max abs diff vs torch is_causal=True:", float(np.abs(out - ref[0].numpy()).max()))
