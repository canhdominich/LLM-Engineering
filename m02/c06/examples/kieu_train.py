import csv
import math
import time

import matplotlib.pyplot as plt
import torch

from minigpt import GPT, OUT, CharTokenizer, GPTConfig, load_kieu

torch.manual_seed(1337)
torch.set_num_threads(1)
STEPS, BATCH, EVAL_EVERY, LR, WARMUP = 1500, 32, 150, 3e-3, 100

text = load_kieu()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text))
n_train = int(0.9 * len(data))
splits = {"train": data[:n_train], "val": data[n_train:]}
cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=4, n_head=4, n_embd=64, dropout=0.1)
model = GPT(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)


def get_batch(split, gen):
    src = splits[split]
    ix = torch.randint(len(src) - cfg.block_size - 1, (BATCH,), generator=gen)
    x = torch.stack([src[i:i + cfg.block_size] for i in ix])
    y = torch.stack([src[i + 1:i + 1 + cfg.block_size] for i in ix])
    return x, y


@torch.no_grad()
def estimate(split):
    model.eval()
    gen = torch.Generator().manual_seed(0)
    losses = [model(*get_batch(split, gen))[1].item() for _ in range(10)]
    model.train()
    return sum(losses) / len(losses)


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / (STEPS - WARMUP)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


print(f"chars train/val: {n_train}/{len(data) - n_train}, vocab {tok.vocab_size}, "
      f"params {sum(p.numel() for p in model.parameters())}, ln(vocab) = {math.log(tok.vocab_size):.3f}")
print(f"{'step':>5} {'train':>6} {'val':>6} {'ppl_val':>7}")
gen, rows, samples, SHOW = torch.Generator().manual_seed(1), [], [], (0, 300, STEPS)
t0, c0 = time.perf_counter(), time.process_time()
for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0:
        tr, va = estimate("train"), estimate("val")
        rows.append((step, round(tr, 4), round(va, 4)))
        print(f"{step:>5} {tr:>6.3f} {va:>6.3f} {math.exp(va):>7.2f}")
    if step in SHOW:
        model.eval()
        out = model.generate(torch.tensor([tok.encode("Trăm năm")]), 40, temperature=0.8, generator=torch.Generator().manual_seed(7))
        model.train()
        samples.append(f"step {step:>4}: " + tok.decode(out[0].tolist()).replace(chr(10), " / "))
    if step == STEPS:
        break
    for group in opt.param_groups:
        group["lr"] = lr_at(step)
    _, loss = model(*get_batch("train", gen))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

OUT.mkdir(exist_ok=True)
with open(OUT / "loss_curve.csv", "w", newline="") as f:
    csv.writer(f).writerows([("step", "train_loss", "val_loss"), *rows])
torch.save({"config": vars(cfg), "model": model.state_dict()}, OUT / "kieu_gpt.pt")
plt.plot([r[0] for r in rows], [r[1] for r in rows], label="train")
plt.plot([r[0] for r in rows], [r[2] for r in rows], label="val")
plt.axhline(math.log(tok.vocab_size), ls=":", c="gray", label="uniform guess")
plt.xlabel("step"), plt.ylabel("cross-entropy (nats per char)"), plt.legend()
plt.savefig(OUT / "loss_curve.png", dpi=120)
print(*samples, sep="\n")
print(f"best val {min(r[2] for r in rows):.3f} at step {min(rows, key=lambda r: r[2])[0]}")
print(f"cpu time {time.process_time() - c0:.0f} s, wall time {time.perf_counter() - t0:.0f} s (1 thread)")
