from collections import Counter

import numpy as np

from _text import load_messages, tokenize

docs = [tokenize(m["text"]) for m in load_messages()]
freq = Counter(w for d in docs for w in d)
vocab = sorted(freq, key=lambda w: (-freq[w], w))
index = {w: i for i, w in enumerate(vocab)}
V, WINDOW = len(vocab), 2

C = np.zeros((V, V))
for doc in docs:
    ids = [index[w] for w in doc]
    for i, center in enumerate(ids):
        for j in range(max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)):
            if j != i:
                C[center, ids[j]] += 1

# PPMI: log của "gặp nhau thật" chia "gặp nhau nếu ngẫu nhiên", số âm đưa về 0
total = C.sum()
expected = np.outer(C.sum(axis=1), C.sum(axis=0)) / total
with np.errstate(divide="ignore"):
    ppmi = np.maximum(np.log(C / expected), 0.0)


def cosine(M, a, b):
    x, y = M[index[a]], M[index[b]]
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


row = C[index["trễ"]]
print("C shape:", C.shape, "| non-zero cells:", f"{(C > 0).mean():.1%}")
print("top contexts of 'trễ' (count):", [(vocab[j], int(row[j])) for j in np.argsort(-row)[:5]])
cols = [index[c] for c in ["giao", "đến", "hơn"]]
for w in ["trễ", "chậm", "địa_chỉ"]:
    print(f"{w:>8} next to giao/đến/hơn: counts {C[index[w], cols].astype(int).tolist()}, ppmi {ppmi[index[w], cols].round(2).tolist()}")
print("pair              raw   ppmi")
for a, b in [("trễ", "chậm"), ("trễ", "muộn"), ("hủy", "huỷ"), ("trễ", "sớm"), ("trễ", "địa_chỉ"), ("trễ", "gạo")]:
    print(f"{a:>5} vs {b:<8} {cosine(C, a, b):.3f}  {cosine(ppmi, a, b):.3f}")
