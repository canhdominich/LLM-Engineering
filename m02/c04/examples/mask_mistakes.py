import numpy as np

rng = np.random.default_rng(5)
n, d = 4, 8
Q, K = rng.normal(size=(n, d)), rng.normal(size=(n, d))
s = Q @ K.T / np.sqrt(d)
future = np.triu(np.ones((n, n), dtype=bool), k=1)


def softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


right = softmax(np.where(future, -np.inf, s))
zero_fill = softmax(np.where(future, 0.0, s))           # sai 1: điền 0 thay vì -inf
after = softmax(s) * ~future                            # sai 2: che sau softmax
print("row 0, right       :", right[0].round(3), "sum", right[0].sum().round(3))
print("row 0, fill with 0 :", zero_fill[0].round(3), "sum", zero_fill[0].sum().round(3))
print("row 0, mask after  :", after[0].round(3), "sum", after[0].sum().round(3))

padding = np.array([True, True, True, True])            # sai 3: một hàng bị che hết
with np.errstate(invalid="ignore"):
    all_masked = softmax(np.where(padding, -np.inf, s[0]))
print("row fully masked   :", all_masked)
