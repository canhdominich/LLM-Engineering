import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=256):
        super().__init__()
        self.h = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)  # Q, K, V gộp trong một phép nhân
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def attn(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)  # (B, h, T, D/h)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # nhánh 1: trộn thông tin giữa các token
        x = x + self.ff(self.ln2(x))  # nhánh 2: xử lý riêng từng token
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    block = Block()
    x = torch.randn(2, 6, 64)  # 2 tin nhắn, 6 token, mỗi token 64 chiều
    print("input :", tuple(x.shape))
    print("after ln1 :", tuple(block.ln1(x).shape))
    print("after attn:", tuple(block.attn(block.ln1(x)).shape))
    print("hidden ff :", tuple(block.ff[0](x).shape))
    print("output:", tuple(block(x).shape))

    for name, part in [("attention", [block.qkv, block.proj]), ("mlp", [block.ff]), ("layernorm", [block.ln1, block.ln2])]:
        print(f"{name:<9} params:", sum(p.numel() for m in part for p in m.parameters()))
    total = sum(p.numel() for p in block.parameters())
    d, f = 64, 256
    formula = (4 * d * d + 4 * d) + (2 * d * f + f + d) + 4 * d
    print("total params:", total, "| formula:", formula, "| 12d^2 + 13d:", 12 * d * d + 13 * d)
