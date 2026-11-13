import hashlib
import random
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from _anhti_text import HANDBOOK, PREF_PAIRS, SFT_PAIRS


def template(question, answer=""):
    return f"Hỏi: {question}\nĐáp: {answer}"


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


def all_text():
    parts = HANDBOOK + [template(q, a) for q, a in SFT_PAIRS] + [template(q, c) + r for q, c, r in PREF_PAIRS]
    return "\n".join(parts) + "\n"


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 96
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.out = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def setup(seed=0):
    torch.set_num_threads(2)
    random.seed(seed)
    torch.manual_seed(seed)
    tok = CharTokenizer(all_text())
    model = TinyGPT(GPTConfig(vocab_size=tok.vocab_size))
    return tok, model


def pretrain(model, tok, steps=800, batch=16, lr=3e-3, seed=0):
    rnd = random.Random(seed)
    stream = []
    for _ in range(60):
        lines = HANDBOOK[:]
        rnd.shuffle(lines)
        stream += tok.encode("\n".join(lines) + "\n")
    data = torch.tensor(stream)
    T = 64
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for step in range(steps):
        ix = torch.randint(0, len(data) - T - 1, (batch,), generator=g)
        x = torch.stack([data[i:i + T] for i in ix])
        y = torch.stack([data[i + 1:i + T + 1] for i in ix])
        loss = F.cross_entropy(model(x).reshape(-1, tok.vocab_size), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


def sft_example(tok, question, answer, block, mask_prompt=True):
    prompt = tok.encode(template(question))
    full = prompt + tok.encode(answer + "\n")
    x = full[:-1]
    y = full[1:]
    # nhãn -100 nghĩa là bỏ qua vị trí đó khi tính loss
    labels = [-100 if mask_prompt and i < len(prompt) - 1 else t for i, t in enumerate(y)]
    pad = block - len(x)
    return x + [0] * pad, labels + [-100] * pad


def sft(model, tok, pairs=SFT_PAIRS, steps=300, lr=1e-3, mask_prompt=True, params=None):
    T = model.cfg.block_size
    rows = [sft_example(tok, q, a, T, mask_prompt) for q, a in pairs]
    x = torch.tensor([r[0] for r in rows])
    y = torch.tensor([r[1] for r in rows])
    opt = torch.optim.AdamW(params if params is not None else [p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        loss = F.cross_entropy(model(x).reshape(-1, tok.vocab_size), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


@torch.no_grad()
def generate(model, tok, prompt, max_new=60):
    ids = tok.encode(prompt)
    for _ in range(max_new):
        logits = model(torch.tensor([ids[-model.cfg.block_size:]]))[0, -1]
        nxt = int(logits.argmax())
        if tok.chars[nxt] == "\n":
            break
        ids.append(nxt)
    return tok.decode(ids[len(tok.encode(prompt)):])


def answer(model, tok, question):
    return generate(model, tok, template(question)).strip()


def answer_logprob(model, tok, question, text):
    prompt = tok.encode(template(question))
    full = prompt + tok.encode(text + "\n")
    x = torch.tensor([full[:-1]])
    logp = F.log_softmax(model(x), dim=-1)[0]
    target = torch.tensor(full[1:])
    picked = logp[torch.arange(len(target)), target]
    return picked[len(prompt) - 1:].sum()


HERE = Path(__file__).resolve().parent
KEY = hashlib.sha256((HERE / "_tinygpt.py").read_bytes() + (HERE / "_anhti_text.py").read_bytes()).hexdigest()[:10]


def cached(stage):
    path = HERE / "_cache" / f"{stage}-{KEY}.pt"
    if stage == "pretrain":
        tok, model = setup()
    else:
        tok, model, _ = cached("pretrain")
    if path.exists():
        model.load_state_dict(torch.load(path))
        model.eval()
        return tok, model, None
    losses = pretrain(model, tok) if stage == "pretrain" else sft(model, tok)
    path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return tok, model, losses
