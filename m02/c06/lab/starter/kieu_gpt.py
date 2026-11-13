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
    # TODO 1: chọn batch_size vị trí ngẫu nhiên (dùng generator), cắt x dài block_size
    # và y là x dịch sang phải một ký tự. Trả về (x, y), mỗi cái shape (batch_size, block_size).
    raise NotImplementedError("TODO 1")


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
        # TODO 2: tính điểm q @ k^T chia căn bậc hai của hd, che tương lai bằng self.mask[:T, :T]
        # (vị trí False nhận -inf), softmax theo trục cuối, dropout, nhân với v,
        # ghép các head về (B, T, C) rồi đi qua self.proj.
        raise NotImplementedError("TODO 2")


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
        # TODO 3: cộng embedding token và embedding vị trí, dropout, đi qua từng block,
        # LayerNorm cuối, lm_head ra logits (B, T, vocab). Nếu có targets, loss là
        # cross-entropy trên logits trải phẳng (B * T, vocab). Trả về (logits, loss).
        raise NotImplementedError("TODO 3")


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    # TODO 4: temperature 0 trả về argmax (shape (B, 1)). Ngược lại chia logits cho temperature,
    # nếu có top_k thì gán -inf cho mọi logit nhỏ hơn logit lớn thứ k, softmax,
    # nếu có top_p thì chỉ giữ tập token xác suất cao nhất có tổng vừa đạt top_p rồi chuẩn hóa lại,
    # cuối cùng lấy mẫu bằng torch.multinomial(probs, 1, generator=generator).
    raise NotImplementedError("TODO 4")


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, generator=None):
    model.eval()
    # TODO 5: lặp max_new_tokens lần: chỉ đưa vào model block_size token cuối,
    # lấy logits của vị trí cuối, gọi sample_next, nối token mới vào idx. Trả về idx.
    raise NotImplementedError("TODO 5")


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
