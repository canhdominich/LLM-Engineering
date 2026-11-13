import csv
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(os.environ.get("KIEU_TEXT", Path(__file__).resolve().parents[2] / "data" / "kieu.txt"))


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
class Config:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1


def get_batch(data, block_size, batch_size, generator=None):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)))

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, hd).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = self.drop(att.softmax(dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


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
        self.lm_head.weight = self.tok_emb.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(dim=-1)
    if top_p is not None:
        sorted_p, order = probs.sort(dim=-1, descending=True)
        keep = (sorted_p.cumsum(dim=-1) - sorted_p) < top_p
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_p * keep)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, generator=None):
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        next_id = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


@torch.no_grad()
def estimate_loss(model, data, block_size, batches=10, batch_size=32):
    model.eval()
    gen = torch.Generator().manual_seed(0)
    losses = [model(*get_batch(data, block_size, batch_size, gen))[1].item() for _ in range(batches)]
    model.train()
    return sum(losses) / len(losses)


def train(model, train_data, val_data, steps, lr=3e-3, batch_size=32, eval_every=150, seed=1):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    gen, history, warmup = torch.Generator().manual_seed(seed), [], min(100, steps // 10 + 1)
    model.train()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            history.append((step, estimate_loss(model, train_data, model.cfg.block_size),
                            estimate_loss(model, val_data, model.cfg.block_size)))
        if step == steps:
            break
        progress = max(0.0, (step - warmup) / max(1, steps - warmup))
        lr_now = lr * (step + 1) / warmup if step < warmup else lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in opt.param_groups:
            group["lr"] = lr_now
        _, loss = model(*get_batch(train_data, model.cfg.block_size, batch_size, gen))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return history


def real_syllable_rate(sample, known):
    words = re.findall(r"\w+", sample.lower())
    return sum(w in known for w in words) / max(1, len(words))


def main():
    torch.manual_seed(1337)
    torch.set_num_threads(1)
    steps = int(os.environ.get("KIEU_STEPS", "1500"))
    text = DATA.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text))
    n = int(0.9 * len(data))
    model = GPT(Config(vocab_size=tok.vocab_size))
    t0 = time.process_time()
    history = train(model, data[:n], data[n:], steps)
    for step, tr, va in history:
        print(f"step {step:>4} train {tr:.3f} val {va:.3f}")
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/loss_curve.csv", "w", newline="") as f:
        csv.writer(f).writerows([("step", "train_loss", "val_loss"), *history])
    known = set(re.findall(r"\w+", text.lower()))
    prompt = torch.tensor([tok.encode("Trăm năm")])
    for temperature in (0.5, 0.8, 1.2):
        out = generate(model, prompt, 80, temperature=temperature, generator=torch.Generator().manual_seed(7))
        sample = tok.decode(out[0].tolist())
        print(f"T={temperature} real syllables {real_syllable_rate(sample, known):.0%}: {sample.replace(chr(10), ' / ')}")
    print(f"cpu time {time.process_time() - t0:.0f} s")


if __name__ == "__main__":
    main()
