import re

import torch

from minigpt import load_checkpoint, load_kieu

model, tok = load_checkpoint()
known = set(re.findall(r"\w+", load_kieu().lower()))
prompt = torch.tensor([tok.encode("Trăm năm")] * 5)
strategies = {
    "greedy": dict(temperature=0),
    "T=0.5": dict(temperature=0.5),
    "T=1.0": dict(temperature=1.0),
    "T=1.5": dict(temperature=1.5),
    "T=1.0 top_k=5": dict(temperature=1.0, top_k=5),
    "T=1.0 top_p=0.9": dict(temperature=1.0, top_p=0.9),
}
print(f"{'strategy':<16} {'real':>5} {'repeat':>6}  first sample")
for name, kw in strategies.items():
    out = model.generate(prompt, 200, generator=torch.Generator().manual_seed(7), **kw)
    texts = [tok.decode(row.tolist()) for row in out]
    words = [re.findall(r"\w+", t.lower()) for t in texts]
    real = sum(w in known for ws in words for w in ws) / sum(len(ws) for ws in words)
    # tỉ lệ cụm 3 âm tiết đã xuất hiện trước đó trong cùng mẫu
    grams = [[tuple(ws[i:i + 3]) for i in range(len(ws) - 2)] for ws in words]
    repeat = sum(len(g) - len(set(g)) for g in grams) / sum(len(g) for g in grams)
    print(f"{name:<16} {real:>5.0%} {repeat:>6.0%}  {texts[0][:62].replace(chr(10), ' / ')}")
