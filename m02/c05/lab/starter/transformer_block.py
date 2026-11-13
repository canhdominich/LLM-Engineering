import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_positions(seq_len, d_model, base=10000.0):
    # TODO 1: trả về tensor (seq_len, d_model): cột chẵn là sin, cột lẻ là cos của pos / base^(i / d_model),
    # với i = 0, 2, 4, ...; báo ValueError khi d_model lẻ.
    raise NotImplementedError


def causal_mask(seq_len):
    # TODO 2: trả về ma trận bool (seq_len, seq_len), True ở những ô token hàng i được nhìn (cột j <= i).
    raise NotImplementedError


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False):
        # TODO 3: x có shape (B, T, D). Tính q, k, v bằng self.qkv (thứ tự q, k, v; mỗi phần D cột),
        # tách thành n_heads head có shape (B, h, T, D / h), tính softmax(q k^T / sqrt(D / h)),
        # che tương lai bằng causal_mask khi causal=True, nhân với v, ghép các head lại rồi qua self.proj.
        raise NotImplementedError


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # TODO 4: up, rồi GELU, rồi down; mỗi token được xử lý riêng.
        raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, causal=False):
        # TODO 5: khối pre-norm có hai đường tắt (residual):
        # x cộng attention của ln1(x), rồi x cộng feed-forward của ln2(x).
        raise NotImplementedError


def block_param_formula(d_model, d_ff):
    # TODO 6: trả về số tham số của TransformerBlock tính bằng công thức
    # (attention có bias, feed-forward có bias, hai LayerNorm có gamma và beta).
    raise NotImplementedError


def count_params(module):
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, n_heads, d_ff = 128, 4, 512
    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(2, 10, d_model) + sinusoidal_positions(10, d_model)
    y = block(x, causal=True)
    ref = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("params      :", count_params(block))
    print("formula     :", block_param_formula(d_model, d_ff))
    print("torch layer :", count_params(ref))
    x2 = x.clone()
    x2[:, -1] = 0.0
    same_past = torch.allclose(block(x, causal=True)[:, :-1], block(x2, causal=True)[:, :-1])
    print("past unchanged when last token changes:", same_past)
