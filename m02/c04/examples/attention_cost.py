import time

import numpy as np

rng = np.random.default_rng(0)


def attention_matrix(X):
    s = X @ X.T / np.sqrt(X.shape[1])
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def attention_loop(X):
    n, d = X.shape
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = X[i] @ X[j] / np.sqrt(d)
        A[i] = np.exp(A[i] - A[i].max())
        A[i] /= A[i].sum()
    return A


X = rng.normal(size=(256, 64))
t0 = time.perf_counter(); A1 = attention_loop(X); t1 = time.perf_counter()
A2 = attention_matrix(X); t2 = time.perf_counter()
print(f"n=256: loop {t1 - t0:.3f}s, matrix {t2 - t1:.4f}s, same result: {np.allclose(A1, A2)}")

print("score matrix size for one head, float32:")
for n in (512, 4096, 32768, 131072):
    print(f"  n={n:6d}: {n * n:>14,d} scores = {n * n * 4 / 2**20:10,.0f} MiB")
