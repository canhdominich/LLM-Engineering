import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "support_corpus.csv"
COMPOUNDS = ["tài xế", "hoàn tiền", "địa chỉ", "giờ hẹn", "kẹt xe", "nước mắm", "bảo vệ", "lề mề",
             "tạp hóa cô tấm", "đầu phố", "cầu gỗ", "xóm vui", "bến sông", "bãi biển", "chợ mới",
             "đồi thông", "khu công nghiệp"]
GROUPS = [["trễ", "chậm", "muộn"], ["hủy", "huỷ"], ["shipper", "tài_xế"], ["gạo", "sữa", "mì", "bánh", "nước_mắm"]]


def tokenize(text):
    text = unicodedata.normalize("NFC", text).lower()
    for phrase in COMPOUNDS:
        text = text.replace(phrase, phrase.replace(" ", "_"))
    return re.findall(r"\b[^\W\d]+\b", text)


def build_vocab(docs, min_count=1):
    freq = Counter(w for d in docs for w in d)
    vocab = sorted((w for w in freq if freq[w] >= min_count), key=lambda w: (-freq[w], w))
    return vocab, np.array([freq[w] for w in vocab], dtype=float)


def make_pairs(docs_ids, window=2):
    pairs = []
    for ids in docs_ids:
        for i, center in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((center, ids[j]))
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def noise_distribution(counts, power=0.75):
    weights = np.asarray(counts, dtype=float) ** power
    return weights / weights.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sgns_grads(v, u_pos, u_neg):
    s_pos = sigmoid((v * u_pos).sum(axis=1))
    s_neg = sigmoid(np.einsum("bkd,bd->bk", u_neg, v))
    losses = -np.log(s_pos + 1e-12) - np.log(1 - s_neg + 1e-12).sum(axis=1)
    g_pos = (s_pos - 1)[:, None]
    g_neg = s_neg[:, :, None]
    grad_v = g_pos * u_pos + (g_neg * u_neg).sum(axis=1)
    grad_pos = g_pos * v
    grad_neg = g_neg * v[:, None, :]
    return losses, grad_v, grad_pos, grad_neg


def train(docs_ids, counts, dim=16, window=2, k=5, epochs=3, lr=0.025, batch=256, seed=0):
    rng = np.random.default_rng(seed)
    V = len(counts)
    W_in = (rng.random((V, dim)) - 0.5) / dim
    W_out = np.zeros((V, dim))
    noise = noise_distribution(counts)
    pairs = make_pairs(docs_ids, window)
    history = []
    for _ in range(epochs):
        rng.shuffle(pairs)
        total = 0.0
        for s in range(0, len(pairs), batch):
            b = pairs[s:s + batch]
            neg = rng.choice(V, size=(len(b), k), p=noise)
            losses, grad_v, grad_pos, grad_neg = sgns_grads(W_in[b[:, 0]], W_out[b[:, 1]], W_out[neg])
            np.add.at(W_in, b[:, 0], -lr * grad_v)
            np.add.at(W_out, b[:, 1], -lr * grad_pos)
            np.add.at(W_out, neg, -lr * grad_neg)
            total += losses.sum()
        history.append(float(total / len(pairs)))
    return W_in, history


def most_similar(W, vocab, word, k=4):
    if word not in vocab:
        raise KeyError(word)
    i = vocab.index(word)
    unit = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    sims = unit @ unit[i]
    sims[i] = -np.inf
    order = np.argsort(-sims, kind="stable")[:k]
    return [(vocab[j], float(sims[j])) for j in order]


def synonym_hit_rate(W, vocab, groups):
    hits = total = 0
    for group in groups:
        for word in group:
            if word not in vocab:
                continue
            top1 = most_similar(W, vocab, word, k=1)[0][0]
            hits += top1 in group
            total += 1
    return hits / total if total else 0.0


def load_docs(path=DATA):
    with open(path, encoding="utf-8") as f:
        return [tokenize(row["text"]) for row in csv.DictReader(f)]


if __name__ == "__main__":
    docs = load_docs()
    vocab, counts = build_vocab(docs, min_count=1)
    index = {w: i for i, w in enumerate(vocab)}
    docs_ids = [[index[w] for w in d] for d in docs]
    W, history = train(docs_ids, counts)
    print("vocabulary:", len(vocab), "| pairs per epoch:", len(make_pairs(docs_ids)))
    print("loss per epoch:", [round(x, 3) for x in history])
    for word in ["trễ", "hủy", "shipper", "đường", "lề_mề"]:
        print(f"{word:>8}:", ", ".join(f"{w} {s:.2f}" for w, s in most_similar(W, vocab, word)))
    print(f"synonym hit rate: {synonym_hit_rate(W, vocab, GROUPS):.3f}")
    top3 = [w for w, _ in most_similar(W, vocab, "trễ", k=3)]
    print("antonym 'sớm' in top-3 of 'trễ':", "sớm" in top3)
