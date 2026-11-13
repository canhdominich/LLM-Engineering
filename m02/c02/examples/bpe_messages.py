import time
import unicodedata
from pathlib import Path

import pandas as pd

from bpe_encode import encode_word
from bpe_train import train_bpe

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
msgs = pd.read_csv(DATA)
train = msgs.loc[msgs["split"] == "train", "text"].tolist()
test = msgs.loc[msgs["split"] == "test", "text"].tolist()


def show(token):
    return "".join(f"<{ord(c):04X}>" if unicodedata.combining(c) else c for c in token)


def tokens_of(text, ranks, form="NFC"):
    text = unicodedata.normalize(form, text.lower())
    return [t for w in text.split() for t in encode_word(w, ranks)]


t0 = time.perf_counter()
merges, _ = train_bpe(train, 300)
print(f"trained {len(merges)} merges in {time.perf_counter() - t0:.2f} s")
ranks = {pair: rank for rank, (pair, _) in enumerate(merges)}
print("first 8:", ["".join(p) for p, _ in merges[:8]])
print("longest:", sorted({"".join(p) for p, _ in merges}, key=lambda s: (len(s), s))[-6:])

for n in (0, 50, 300):
    r = {p: k for p, k in ranks.items() if k < n}
    avg = sum(len(tokens_of(t, r)) for t in test) / len(test)
    print(f"merges={n:3}  tokens/msg on test={avg:5.1f}")

msg = "Đơn hỏa tốc mà trễ 40 phút rồi"
for form in ("NFC", "NFD"):
    toks = tokens_of(msg, ranks, form)
    print(f"{form}: {len(toks):2} tokens", [show(t) for t in toks[:9]])
