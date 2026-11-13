import numpy as np

rng = np.random.default_rng(3)
d = 8
q, k = rng.normal(size=d), rng.normal(size=d)  # query của "huy", key của "don"


def rope(x, pos, base=10000.0):
    theta = pos / base ** (np.arange(0, d, 2) / d)  # mỗi cặp chiều một góc quay
    cos, sin = np.cos(theta), np.sin(theta)
    x1, x2 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x1 * cos - x2 * sin
    out[1::2] = x1 * sin + x2 * cos
    return out


def sinusoidal(pos, base=10000.0):
    angle = pos / base ** (np.arange(0, d, 2) / d)
    pe = np.empty(d)
    pe[0::2], pe[1::2] = np.sin(angle), np.cos(angle)
    return pe


print("no position     :", round(float(q @ k), 4))
for m, n in [(2, 5), (10, 13), (100, 103), (5, 2)]:
    s_rope = rope(q, m) @ rope(k, n)
    s_add = (q + sinusoidal(m)) @ (k + sinusoidal(n))  # cộng vị trí tuyệt đối rồi mới nhân
    print(f"q at {m:>3}, k at {n:>3}: rope = {s_rope:.4f}   added = {s_add:.4f}")

print("norm before/after rope:", round(float(np.linalg.norm(q)), 4), round(float(np.linalg.norm(rope(q, 100))), 4))
