from collections import Counter

import numpy as np

from _text import load_messages, tokenize


def load_corpus(min_count=1):
    docs = [tokenize(m["text"]) for m in load_messages()]
    freq = Counter(w for d in docs for w in d)
    vocab = sorted((w for w in freq if freq[w] >= min_count), key=lambda w: (-freq[w], w))
    index = {w: i for i, w in enumerate(vocab)}
    ids = [[index[w] for w in d if w in index] for d in docs]
    return vocab, index, ids, np.array([freq[w] for w in vocab], dtype=float)


def make_pairs(docs_ids, window=2):
    pairs = []
    for ids in docs_ids:
        for i, center in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((center, ids[j]))
    return np.array(pairs)


def sgns_step(W_in, W_out, centers, contexts, negatives, lr):
    v = W_in[centers]                      # (B, d) vector của từ trung tâm
    u_pos = W_out[contexts]                # (B, d) vector của hàng xóm thật
    u_neg = W_out[negatives]               # (B, k, d) vector của từ nhiễu
    s_pos = 1 / (1 + np.exp(-(v * u_pos).sum(1)))
    s_neg = 1 / (1 + np.exp(-np.einsum("bkd,bd->bk", u_neg, v)))
    loss = -np.log(s_pos + 1e-9).mean() - np.log(1 - s_neg + 1e-9).sum(1).mean()
    g_pos = (s_pos - 1)[:, None]           # kéo hàng xóm thật lại gần
    g_neg = s_neg[:, :, None]              # đẩy từ nhiễu ra xa
    grad_v = g_pos * u_pos + (g_neg * u_neg).sum(1)
    np.add.at(W_out, contexts, -lr * g_pos * v)
    np.add.at(W_out, negatives, -lr * g_neg * v[:, None, :])
    np.add.at(W_in, centers, -lr * grad_v)
    return loss


def train(docs_ids, freq, dim=16, window=2, k=5, epochs=5, lr=0.05, batch=256, seed=0):
    rng = np.random.default_rng(seed)
    V = len(freq)
    W_in = (rng.random((V, dim)) - 0.5) / dim
    W_out = np.zeros((V, dim))
    noise = freq ** 0.75 / (freq ** 0.75).sum()
    pairs = make_pairs(docs_ids, window)
    history = []
    for epoch in range(epochs):
        rng.shuffle(pairs)
        losses = []
        for s in range(0, len(pairs), batch):
            b = pairs[s:s + batch]
            negatives = rng.choice(V, size=(len(b), k), p=noise)
            losses.append(sgns_step(W_in, W_out, b[:, 0], b[:, 1], negatives, lr))
        history.append(float(np.mean(losses)))
    return W_in, history


def neighbors(W, vocab, index, word, k=4):
    unit = W / np.linalg.norm(W, axis=1, keepdims=True)
    sims = unit @ unit[index[word]]
    order = [j for j in np.argsort(-sims) if j != index[word]][:k]
    return [(vocab[j], float(sims[j])) for j in order]
