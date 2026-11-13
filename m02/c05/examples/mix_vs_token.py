import torch

from block_forward import Block

torch.manual_seed(1)
block = Block().eval()
tokens = "huy don khong doi dia chi".split()
x = torch.randn(1, 6, 64)
x2 = x.clone()
x2[0, 5] = torch.randn(64)  # chỉ đổi token cuối: "chi" thành một từ khác

with torch.no_grad():
    a1, a2 = block.attn(block.ln1(x)), block.attn(block.ln1(x2))
    f1, f2 = block.ff(block.ln2(x)), block.ff(block.ln2(x2))

print("token   attn change   mlp change")
for i, t in enumerate(tokens):
    da = (a1[0, i] - a2[0, i]).abs().max().item()
    df = (f1[0, i] - f2[0, i]).abs().max().item()
    print(f"{t:<7} {da:>11.4f} {df:>12.4f}")
