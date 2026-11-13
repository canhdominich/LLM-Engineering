import json
import os
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
req = urllib.request.Request(f"{HOST}/api/show", data=json.dumps({"model": "qwen3:1.7b"}).encode())
info = json.load(urllib.request.urlopen(req, timeout=60))["model_info"]
layers, heads = info["qwen3.block_count"], info["qwen3.attention.head_count"]
kv_heads, head_dim = info["qwen3.attention.head_count_kv"], info["qwen3.attention.key_length"]
BYTES = 2  # fp16 cho mỗi số trong KV cache


def kv_bytes(n_kv_heads, tokens):
    # 2 vì lưu cả K và V, ở mọi lớp, cho mọi token đã thấy
    return 2 * layers * n_kv_heads * head_dim * BYTES * tokens


print(f"qwen3:1.7b layers={layers} q_heads={heads} kv_heads={kv_heads} head_dim={head_dim}")
print(f"KV per token: MHA {kv_bytes(heads, 1) / 1024:.0f} KiB | GQA {kv_bytes(kv_heads, 1) / 1024:.0f} KiB | MQA {kv_bytes(1, 1) / 1024:.0f} KiB")
print(f"{'context':>8} {'MHA (16)':>11} {'GQA (8)':>11} {'MQA (1)':>11}")
for tokens in (2_000, 8_192, 32_768):
    row = [kv_bytes(h, tokens) / 2**30 for h in (heads, kv_heads, 1)]
    print(f"{tokens:>8,} " + " ".join(f"{x:>8.2f} GiB" for x in row))
tags = json.load(urllib.request.urlopen(f"{HOST}/api/tags", timeout=60))["models"]
size = next(m["size"] for m in tags if m["name"] == "qwen3:1.7b")
print(f"weights file (Q4_K_M): {size / 2**30:.2f} GiB")
chats = 20
print(f"{chats} chats x 2,000 tokens with GQA: {kv_bytes(kv_heads, 2_000) * chats / 2**30:.2f} GiB")
