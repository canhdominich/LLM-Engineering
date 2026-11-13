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
    # TODO 1: chuẩn hóa NFC, chữ thường, nối các cụm trong COMPOUNDS bằng "_",
    # rồi lấy các từ chỉ gồm chữ cái (bỏ số, dấu câu, mã đơn như AT012345)
    raise NotImplementedError


def build_vocab(docs, min_count=1):
    # TODO 2: đếm tần suất, giữ từ xuất hiện ít nhất min_count lần,
    # sắp theo tần suất giảm dần rồi theo chữ cái; trả về (vocab, counts dạng np.array float)
    raise NotImplementedError


def make_pairs(docs_ids, window=2):
    # TODO 3: với mỗi tin nhắn, mỗi vị trí i, ghép (ids[i], ids[j]) cho mọi j cách i không quá window, j khác i;
    # không ghép qua ranh giới hai tin nhắn; trả về mảng int shape (n, 2)
    raise NotImplementedError


def noise_distribution(counts, power=0.75):
    # TODO 4: xác suất chọn từ nhiễu tỉ lệ với count ** power, tổng bằng 1
    raise NotImplementedError


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sgns_grads(v, u_pos, u_neg):
    # TODO 5: v (B, d), u_pos (B, d), u_neg (B, k, d)
    # losses (B,) = -log σ(u_pos·v) - Σ log σ(-u_neg·v)
    # trả về losses, grad_v (B, d), grad_pos (B, d), grad_neg (B, k, d) là đạo hàm của losses.sum()
    raise NotImplementedError


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
    # TODO 6: báo KeyError nếu word không có trong vocab; cosine của word với mọi hàng của W;
    # bỏ chính word; trả về k cặp (từ, điểm) điểm giảm dần
    raise NotImplementedError


def synonym_hit_rate(W, vocab, groups):
    # TODO 7: với mỗi từ có trong vocab của từng nhóm, lấy láng giềng top-1;
    # trả về tỉ lệ từ có top-1 nằm cùng nhóm (bỏ qua từ không có trong vocab)
    raise NotImplementedError


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
