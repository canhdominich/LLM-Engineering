from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
df = pd.read_csv(DATA)
train = df[df["split"] == "train"].reset_index(drop=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train["text"])  # ma trận thưa, mỗi hàng dài 1

queries = [
    "khi nào shipper giao tới Cầu Gỗ",
    "tôi muốn lấy lại tiền",
    "toi muon lay lai tien",
    "tôi không muốn hoàn tiền, chỉ muốn đổi địa chỉ",
    "bao lâu nữa thì được hoàn tiền",
]
for q in queries:
    scores = (X @ vectorizer.transform([q]).T).toarray().ravel()  # cosine vì hai vector đã dài 1
    best = scores.argmax()
    print(f"query {q!r}")
    print(f"  best {scores[best]:.3f} [{train.loc[best, 'intent']}] {train.loc[best, 'text'][:52]!r}")
