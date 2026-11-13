import torch

src = "huy don khong doi dia chi".split()  # câu khách gửi (encoder đọc)
tgt = "cancel order keep address".split()  # câu đang sinh (decoder viết)


def show(name, allowed, rows, cols):
    print(name)
    print(" " * 7 + " ".join(f"{c:>6}" for c in cols))
    for r, line in zip(rows, allowed):
        print(f"{r:>6} " + " ".join(f"{'x' if v else '.':>6}" for v in line))


T, S = len(src), len(tgt)
bidirectional = torch.ones(T, T, dtype=torch.bool)  # BERT: mọi token thấy mọi token
causal = torch.tril(torch.ones(S, S, dtype=torch.bool))  # GPT: chỉ thấy mình và bên trái
cross = torch.ones(S, T, dtype=torch.bool)  # T5: mỗi token đang sinh nhìn cả câu nguồn

show("encoder self-attention (BERT)", bidirectional, src, src)
show("decoder self-attention (GPT)", causal, tgt, tgt)
show("cross-attention (T5 decoder reads encoder)", cross, tgt, src)
print("visible pairs:", int(bidirectional.sum()), int(causal.sum()), int(cross.sum()))
