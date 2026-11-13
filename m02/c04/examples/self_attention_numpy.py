import numpy as np

TOKENS = ["don", "AT027075", "giao", "tre", "toi", "muon", "huy", "no"]
rng = np.random.default_rng(7)
n, d_model, d_k = len(TOKENS), 16, 8
X = rng.normal(size=(n, d_model))          # mỗi token một vector, ở đây là số ngẫu nhiên
Wq = rng.normal(size=(d_model, d_k)) / np.sqrt(d_model)
Wk = rng.normal(size=(d_model, d_k)) / np.sqrt(d_model)
Wv = rng.normal(size=(d_model, d_k)) / np.sqrt(d_model)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def self_attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv           # (n, d_k) mỗi ma trận
    scores = Q @ K.T / np.sqrt(K.shape[-1])     # (n, n): token i hỏi, token j trả lời
    A = softmax(scores, axis=-1)                # mỗi hàng cộng lại bằng 1
    return A @ V, A                             # (n, d_k): trung bình có trọng số của V


out, A = self_attention(X, Wq, Wk, Wv)
print("shapes: X", X.shape, " Q, K, V", (X @ Wq).shape, " A", A.shape, " out", out.shape)
print("row sums:", A.sum(axis=1).round(6))
print("weights of token 'no' (row 7):")
for tok, w in zip(TOKENS, A[7]):
    print(f"  {tok:9s} {w:.3f}")
raw = (X @ Wq) @ (X @ Wk).T  # quên chia căn bậc hai của d_k
print("row 7 scores, scaled  :", (raw[7] / np.sqrt(d_k)).round(2))
print("row 7 scores, unscaled:", raw[7].round(2))
print("row 7 without scaling:", softmax(raw, axis=-1)[7].round(3))
manual = (A[7][:, None] * (X @ Wv)).sum(axis=0)
print("out[7] equals weighted sum of V rows:", np.allclose(out[7], manual))
perm = rng.permutation(n)  # xáo thứ tự token
out_perm, _ = self_attention(X[perm], Wq, Wk, Wv)
print("shuffle tokens, outputs only shuffle:", np.allclose(out_perm, out[perm]))
