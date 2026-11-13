import re

import numpy as np

messages = [
    "đã giao, chưa thanh toán",
    "đã thanh toán, chưa giao",
    "đơn chưa giao",
    "da giao, chua thanh toan",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())  # \w nhận cả chữ có dấu tiếng Việt


vocab = sorted({w for m in messages for w in tokenize(m)})
index = {w: i for i, w in enumerate(vocab)}
X = np.zeros((len(messages), len(vocab)), dtype=int)
for row, m in enumerate(messages):
    for w in tokenize(m):
        X[row, index[w]] += 1

print("vocabulary:", vocab)
for m, vec in zip(messages, X):
    print(f"{m!r:28} {vec}")
print("message 0 and 1 have the same vector:", bool((X[0] == X[1]).all()))
print("shared words between message 0 and 3:", int((X[0] * X[3]).sum()))
