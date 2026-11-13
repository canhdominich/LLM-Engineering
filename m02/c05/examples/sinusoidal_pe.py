import numpy as np

np.set_printoptions(precision=3, suppress=True)


def sinusoidal(n, d, base=10000.0):
    pos = np.arange(n)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / base ** (i / d)  # cặp chiều i quay với tần số 1 / base^(i/d)
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe


P = sinusoidal(128, 16)
print("shape:", P.shape)
print("position 0:", P[0][:6])
print("position 1:", P[1][:6])
print("position 2:", P[2][:6])

for p, q in [(2, 5), (10, 13), (100, 103)]:
    print(f"dot(P[{p}], P[{q}]) = {P[p] @ P[q]:.4f}")

for k in [0, 1, 4, 16, 64]:
    print(f"offset {k:>2}: dot = {P[20] @ P[20 + k]:.3f}")

wavelength = 2 * np.pi * 10000.0 ** (np.arange(0, 16, 2) / 16)
print("wavelength per pair (tokens):", wavelength.round(1))
