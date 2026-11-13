import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
msgs = pd.read_csv(DATA)


def clean(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def search_key(text):
    s = unicodedata.normalize("NFD", clean(text).casefold())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")


print(search_key("Đơn  AT027075 ở Khu Công Nghiệp"))

keys = msgs["text"].map(search_key)
exact = msgs["text"].map(clean).str.lower().str.contains(r"\bđơn\b")
loose = keys.str.contains(r"\bdon\b")
print(f"messages mentioning an order  exact='đơn': {exact.sum()}  key='don': {loose.sum()}")

# Bỏ dấu làm những từ khác nghĩa trùng nhau
words = {w for t in msgs.loc[msgs["has_diacritics"], "text"] for w in clean(t).lower().split()}
groups = defaultdict(set)
for w in words:
    groups[search_key(w)].add(w)
clash = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
print(f"distinct words with diacritics: {len(words)}  keys: {len(groups)}  keys shared by 2+ words: {len(clash)}")
for k in sorted(clash)[:6]:
    print(f"  {k}: {clash[k]}")
