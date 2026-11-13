from collections import Counter

import numpy as np
import torch

from _text import load_messages, tokenize

messages = load_messages()
docs = [tokenize(m["text"]) for m in messages]
freq = Counter(w for d in docs for w in d)
vocab = sorted(freq, key=lambda w: (-freq[w], w))
index = {w: i for i, w in enumerate(vocab)}
V, WINDOW, DIM = len(vocab), 2, 16

C = np.zeros((V, V))
for doc in docs:
    ids = [index[w] for w in doc]
    for i, center in enumerate(ids):
        for j in range(max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)):
            if j != i:
                C[center, ids[j]] += 1
expected = np.outer(C.sum(axis=1), C.sum(axis=0)) / C.sum()
with np.errstate(divide="ignore"):
    ppmi = np.maximum(np.log(C / expected), 0.0)

U, S, Vt = np.linalg.svd(ppmi)
E = U[:, :DIM] * S[:DIM]  # mỗi từ còn 16 số thay vì 117
unit = E / np.linalg.norm(E, axis=1, keepdims=True)


def neighbors(word, k=4):
    sims = unit @ unit[index[word]]
    order = [j for j in np.argsort(-sims) if j != index[word]][:k]
    return ", ".join(f"{vocab[j]} {sims[j]:.2f}" for j in order)


print("ppmi:", ppmi.shape, "| dense E:", E.shape, f"| energy kept: {(S[:DIM] ** 2).sum() / (S ** 2).sum():.1%}")
print("dense 'trễ' (first 6 of 16):", np.round(E[index["trễ"], :6], 2))
for word in ["trễ", "hủy", "shipper", "gạo"]:
    print(f"{word:>8}: {neighbors(word)}")

table = torch.nn.Embedding.from_pretrained(torch.tensor(E, dtype=torch.float32))
looked_up = table(torch.tensor([index["trễ"]]))
print("nn.Embedding lookup equals row:", np.allclose(looked_up.numpy()[0], E[index["trễ"]], atol=1e-6))

sims = unit @ unit[index["trễ"]]
keywords = [vocab[j] for j in np.argsort(-sims) if sims[j] > 0.9]
hits = [any(k in d for k in keywords) for d in docs]
late = [m["intent"] == "late" for m in messages]
print("keywords with cosine > 0.9 to 'trễ':", keywords)
print(f"late complaints matched: {sum(h and l for h, l in zip(hits, late))}/{sum(late)}, other messages matched: {sum(h and not l for h, l in zip(hits, late))}")
