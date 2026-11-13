import unicodedata
from pathlib import Path

import pandas as pd
import tiktoken
from transformers import AutoTokenizer

DATA = Path(__file__).resolve().parents[1] / "data" / "support_pairs.csv"
pairs = pd.read_csv(DATA)
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
counters = {
    "o200k_base": tiktoken.get_encoding("o200k_base").encode,
    "cl100k_base": tiktoken.get_encoding("cl100k_base").encode,
    "r50k_base": tiktoken.get_encoding("r50k_base").encode,
    "qwen3": qwen.encode,
}


def strip_accents(text):
    s = unicodedata.normalize("NFD", text)
    return "".join(c for c in s if not unicodedata.combining(c)).replace("đ", "d").replace("Đ", "D")


pairs["vi_plain"] = pairs["vi"].map(strip_accents)
print(f"{len(pairs)} message pairs, characters vi={pairs['vi'].str.len().sum()} en={pairs['en'].str.len().sum()}")
print(f"{'tokenizer':12} {'vi':>5} {'en':>5} {'vi/en':>6} {'vi no-accent':>13}")
for name, encode in counters.items():
    n = {col: sum(len(encode(t)) for t in pairs[col]) for col in ("vi", "en", "vi_plain")}
    print(f"{name:12} {n['vi']:5} {n['en']:5} {n['vi'] / n['en']:6.2f} {n['vi_plain']:13}")
