import unicodedata
from pathlib import Path

import pandas as pd
import tiktoken

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
msgs = pd.read_csv(DATA)
enc = tiktoken.get_encoding("o200k_base")


def nfc(text):
    return unicodedata.normalize("NFC", text)


raw = msgs["text"]
fixed = raw.map(nfc)
changed = raw != fixed
print(f"messages: {len(msgs)}  not NFC: {changed.sum()}")

# Luật từ khóa của bot cũ: tin nhắn có chữ "đơn" thì hỏi mã đơn
rule = "đơn"
hit_raw = raw.str.lower().str.contains(rule, regex=False)
hit_fixed = fixed.str.lower().str.contains(rule, regex=False)
print(f"rule '{rule}' hits  raw={hit_raw.sum()}  after NFC={hit_fixed.sum()}")
print("missed only because of NFD:", (hit_fixed & ~hit_raw).sum())

sample = msgs.loc[changed, "text"].iloc[0]
print("example:", sample)
print(f"  chars {len(sample)} vs {len(nfc(sample))}, tokens {len(enc.encode(sample))} vs {len(enc.encode(nfc(sample)))}")

tok_raw = sum(len(enc.encode(t)) for t in raw[changed])
tok_fixed = sum(len(enc.encode(t)) for t in fixed[changed])
print(f"o200k tokens of the {changed.sum()} NFD messages: {tok_raw} raw vs {tok_fixed} NFC (+{tok_raw / tok_fixed - 1:.0%})")
