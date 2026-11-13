import time

import torch

torch.manual_seed(0)
torch.set_num_threads(1)
W = torch.randn(2048, 2048) / 45  # một ma trận trọng số cỡ một lớp của qwen3:1.7b
X = torch.randn(256, 2048)  # 256 token của prompt


def best_ms(fn, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1000


def all_at_once():
    return X @ W  # prefill: một phép nhân cho cả 256 token


def one_by_one():
    return torch.stack([X[i] @ W for i in range(256)])  # decode: mỗi lần một token


same = torch.allclose(all_at_once(), one_by_one(), atol=1e-4)
a, b = best_ms(all_at_once), best_ms(one_by_one)
print(f"weights read per call: {W.numel() * 4 / 1e6:.1f} MB")
print(f"256 tokens at once : {a:7.2f} ms  ({256 / a * 1000:8.0f} tokens/s)")
print(f"256 tokens one by one: {b:7.2f} ms  ({256 / b * 1000:8.0f} tokens/s)")
print(f"ratio {b / a:.1f}x, same result: {same}")
