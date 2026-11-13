import math
from collections import Counter

import torch

from minigpt import CharTokenizer, load_kieu

text = load_kieu()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text))
n_train = int(0.9 * len(data))
print(f"lines {text.count(chr(10))}, chars {len(text):,}, vocab {tok.vocab_size}")
print(f"train {n_train:,} chars (verses 1-{text[:n_train].count(chr(10)) + 1}), val {len(data) - n_train:,} chars")
print("most common:", [(ch if ch != " " else "space", n) for ch, n in Counter(text).most_common(6)])
counts = torch.bincount(data[:n_train], minlength=tok.vocab_size).float()
p = counts / counts.sum()
unigram = -(torch.bincount(data[n_train:], minlength=tok.vocab_size) * torch.log(p.clamp_min(1e-9))).sum() / (len(data) - n_train)
print(f"loss if guessing uniformly: {math.log(tok.vocab_size):.3f} | loss of char-frequency guess on val: {unigram:.3f}")
x = data[:16]
y = data[1:17]
print("x:", repr(tok.decode(x.tolist())))
print("y:", repr(tok.decode(y.tolist())))
print("first 4 training pairs:", [(tok.decode(x[:t + 1].tolist()), tok.decode([y[t].item()])) for t in range(4)])
