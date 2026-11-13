import statistics
import time

import torch

from _tiny_gpt import make_model

model = make_model()
prompt = torch.randint(0, 512, (1, 32), generator=torch.Generator().manual_seed(1))
NEW = 224


@torch.no_grad()
def generate_no_cache(idx):
    times, rows = [], 0
    for _ in range(NEW):
        t0 = time.perf_counter()
        logits = model(idx)  # chạy lại toàn bộ chuỗi mỗi bước
        rows += idx.shape[1]
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        times.append(time.perf_counter() - t0)
    return idx, times, rows


@torch.no_grad()
def generate_with_cache(idx):
    caches = [{} for _ in model.blocks]
    logits = model(idx, caches)  # prefill: cả prompt một lần
    times, rows = [], idx.shape[1]
    for _ in range(NEW):
        t0 = time.perf_counter()
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        logits = model(nxt, caches, start=idx.shape[1] - 1)  # decode: chỉ token mới
        rows += 1
        times.append(time.perf_counter() - t0)
    return idx, times, rows, caches


out_a, t_a, rows_a = generate_no_cache(prompt)
out_b, t_b, rows_b, caches = generate_with_cache(prompt)
print(f"prompt 32 tokens, generate {NEW}, same tokens: {torch.equal(out_a, out_b)}")
print(f"token rows computed: no cache {rows_a:,}, cache {rows_b:,}")
print(f"{'steps':>9} {'no cache ms':>12} {'cache ms':>9}  (median per step)")
for lo in (0, 96, 192):
    a = statistics.median(t_a[lo:lo + 32]) * 1000
    b = statistics.median(t_b[lo:lo + 32]) * 1000
    print(f"{lo:>4}-{lo + 31:<4} {a:>12.2f} {b:>9.2f}")
print(f"total no cache  : {sum(t_a):6.2f} s")
print(f"total with cache: {sum(t_b):6.2f} s  speedup {sum(t_a) / sum(t_b):.1f}x")
kv_bytes = sum(c["k"].numel() + c["v"].numel() for c in caches) * 4
print(f"kv cache at {out_b.shape[1]} tokens: {kv_bytes / 1e6:.2f} MB float32")
