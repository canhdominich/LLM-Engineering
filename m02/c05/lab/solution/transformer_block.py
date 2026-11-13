import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_positions(seq_len, d_model, base=10000.0):
    if d_model % 2:
        raise ValueError("d_model must be even")
    pos = torch.arange(seq_len, dtype=torch.float32)[:, None]
    i = torch.arange(0, d_model, 2, dtype=torch.float32)[None, :]
    angle = pos / base ** (i / d_model)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False):
        B, T, D = x.shape
        hd = D // self.n_heads
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, hd).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
        if causal:
            scores = scores.masked_fill(~causal_mask(T).to(x.device), float("-inf"))
        weights = scores.softmax(dim=-1)
        y = (weights @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, causal=False):
        x = x + self.attn(self.ln1(x), causal=causal)
        x = x + self.ff(self.ln2(x))
        return x


def block_param_formula(d_model, d_ff):
    attention = 4 * d_model * d_model + 4 * d_model
    mlp = 2 * d_model * d_ff + d_ff + d_model
    norms = 4 * d_model
    return attention + mlp + norms


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
