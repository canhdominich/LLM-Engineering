import numpy as np
import torch

rng = np.random.default_rng(3)
n, d_model, n_heads = 8, 16, 4
d_head = d_model // n_heads
X = rng.normal(size=(n, d_model))
Wq, Wk, Wv, Wo = (rng.normal(size=(d_model, d_model)) / np.sqrt(d_model) for _ in range(4))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def split_heads(M):
    return M.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (n, d_model) thành (h, n, d_head)


def multi_head(X):
    Q, K, V = split_heads(X @ Wq), split_heads(X @ Wk), split_heads(X @ Wv)
    A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(d_head))   # (h, n, n): mỗi đầu một bảng
    heads = A @ V                                              # (h, n, d_head)
    concat = heads.transpose(1, 0, 2).reshape(n, d_model)      # ghép các đầu lại
    return concat @ Wo, A


out, A = multi_head(X)
print("Q after split:", split_heads(X @ Wq).shape, " A:", A.shape, " out:", out.shape)
print("token 7 looks most at, per head:", A[:, 7].argmax(axis=1))

mha = torch.nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True, dtype=torch.float64)
with torch.no_grad():
    mha.in_proj_weight.copy_(torch.from_numpy(np.concatenate([Wq.T, Wk.T, Wv.T])))  # torch lưu W dạng chuyển vị
    mha.out_proj.weight.copy_(torch.from_numpy(Wo.T))
    x = torch.from_numpy(X)[None]
    ref, _ = mha(x, x, x, need_weights=False)
print("max abs diff vs torch.nn.MultiheadAttention:", float(np.abs(out - ref[0].numpy()).max()))
print("parameters: ours", 4 * d_model * d_model, " torch", sum(p.numel() for p in mha.parameters()))
