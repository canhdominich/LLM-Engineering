import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(7)
n, d_model, d_k = 8, 16, 8
X = rng.normal(size=(n, d_model))
Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) / np.sqrt(d_model) for _ in range(3))


def self_attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(d_k)
    A = np.exp(scores - scores.max(axis=-1, keepdims=True))
    A = A / A.sum(axis=-1, keepdims=True)
    return A @ V


ours = self_attention(X, Wq, Wk, Wv)
Q, K, V = (torch.from_numpy(X @ W) for W in (Wq, Wk, Wv))
ref = F.scaled_dot_product_attention(Q[None], K[None], V[None])[0].numpy()  # thêm trục batch
print("torch", torch.__version__, " output shape:", ref.shape)
print("max abs difference:", float(np.abs(ours - ref).max()))
print("allclose:", np.allclose(ours, ref, atol=1e-10))
