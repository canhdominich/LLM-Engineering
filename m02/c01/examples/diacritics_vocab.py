import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
train = pd.read_csv(DATA).query("split == 'train'")["text"]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")  # bỏ dấu thanh, dấu mũ
    text = text.replace("đ", "d")  # đ không tách được thành d + dấu nên phải thay riêng
    return re.sub(r"\bat\d{6}\b", "ORDERID", text)


variants = {
    "raw text": TfidfVectorizer(),
    "strip_accents='unicode'": TfidfVectorizer(strip_accents="unicode"),
    "normalize()": TfidfVectorizer(preprocessor=normalize),
}
for name, vec in variants.items():
    vocab = vec.fit(train).vocabulary_
    has = [w for w in ("hoàn", "hoan", "đơn", "đon", "don") if w in vocab]
    print(f"{name:24} vocabulary {len(vocab):3}  contains {has}")
print(normalize("Đơn AT036959 trễ quá, hoàn tiền giúp tôi"))
print("tôi, tới, tối all become:", {normalize(w) for w in ("tôi", "tới", "tối")})
