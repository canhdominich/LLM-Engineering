from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
df = pd.read_csv(DATA)
train = df.loc[df["split"] == "train", "text"].tolist()
n = len(train)

counts = CountVectorizer().fit(train)
C = counts.transform(train)
doc_freq = np.asarray((C > 0).sum(axis=0)).ravel()  # số tin nhắn chứa mỗi từ
vocab = counts.vocabulary_

print(f"training messages: {n}, vocabulary: {len(vocab)} words")
print(f"{'word':8} {'df':>4} {'idf':>6}")
for w in ["đơn", "giao", "tôi", "tiền", "trễ", "muốn", "hoàn", "hoan"]:
    df_w = doc_freq[vocab[w]]
    idf = np.log((1 + n) / (1 + df_w)) + 1  # công thức mặc định của scikit-learn
    print(f"{w:8} {df_w:>4} {idf:>6.3f}")

tfidf = TfidfVectorizer().fit(train)
message = "Tôi muốn hoàn tiền đơn AT000001"
words = [w for w in counts.build_analyzer()(message) if w in vocab]  # từ lạ như at000001 bị bỏ
idf = np.log((1 + n) / (1 + doc_freq[[vocab[w] for w in words]])) + 1
by_hand = idf / np.linalg.norm(idf)  # mỗi từ xuất hiện 1 lần nên tf = 1
row = tfidf.transform([message]).toarray()[0]
print(f"weights for {message!r}:")
for w, h in sorted(zip(words, by_hand), key=lambda x: -x[1]):
    print(f"  {w:6} by hand {h:.3f}   sklearn {row[tfidf.vocabulary_[w]]:.3f}")

plain = "toi muon hoan tien don at000001"
other = tfidf.transform([plain]).toarray()[0]
shown = ", ".join(f"{w} {other[tfidf.vocabulary_[w]]:.3f}" for w in plain.split() if w in tfidf.vocabulary_)
print(f"weights for {plain!r}: {shown}")
print(f"cosine between the two messages: {row @ other:.3f}")
