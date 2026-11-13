from collections import Counter

import numpy as np

from _text import load_messages, tokenize

docs = [tokenize(m["text"]) for m in load_messages()]
freq = Counter(w for d in docs for w in d)
vocab = sorted(freq, key=lambda w: (-freq[w], w))
index = {w: i for i, w in enumerate(vocab)}


def one_hot(word):
    v = np.zeros(len(vocab))
    v[index[word]] = 1.0
    return v


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


late = one_hot("trễ")
print("vocabulary size:", len(vocab), "| index of 'trễ':", index["trễ"])
print("one-hot 'trễ' shape:", late.shape, "| non-zero:", int(late.sum()))
for a, b in [("trễ", "chậm"), ("trễ", "muộn"), ("hủy", "huỷ"), ("trễ", "gạo")]:
    va, vb = one_hot(a), one_hot(b)
    print(f"{a:>5} vs {b:<5} cosine={cosine(va, vb):.3f} distance={np.linalg.norm(va - vb):.3f}")
