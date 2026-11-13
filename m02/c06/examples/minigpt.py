from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parents[1] / "data" / "kieu.txt"
OUT = Path(__file__).resolve().parents[1] / "outputs"


def load_kieu():
    return DATA.read_text(encoding="utf-8")


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


@dataclass
class GPTConfig:
    vocab_size: int = 132
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1
    causal: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head, self.causal, self.dropout = cfg.n_head, cfg.causal, cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) thành (B, n_head, T, C // n_head): mỗi head một góc nhìn
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=p)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # dùng chung ma trận: embedding vào và lớp dự đoán ra
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, generator=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :]
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = filter_probs(logits / temperature, top_k, top_p)
                next_id = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def filter_probs(logits, top_k=None, top_p=None):
    if top_k is not None:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p is not None:
        sorted_p, order = probs.sort(dim=-1, descending=True)
        # giữ token khi tổng xác suất của các token đứng trước nó còn nhỏ hơn top_p
        keep = (sorted_p.cumsum(dim=-1) - sorted_p) < top_p
        sorted_p = sorted_p * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_p)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def load_checkpoint(name="kieu_gpt.pt"):
    ckpt = torch.load(OUT / name, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, CharTokenizer(load_kieu())


if __name__ == "__main__":
    torch.manual_seed(0)
    tok = CharTokenizer(load_kieu())
    model = GPT(GPTConfig(vocab_size=tok.vocab_size))
    idx = torch.tensor([tok.encode("Trăm năm trong cõi")])
    logits, _ = model(idx)
    print("vocab_size:", tok.vocab_size)
    print("input shape:", tuple(idx.shape), "logits shape:", tuple(logits.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
