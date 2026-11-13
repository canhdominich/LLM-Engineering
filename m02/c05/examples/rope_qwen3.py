import json
import os
import urllib.request

import numpy as np

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
req = urllib.request.Request(HOST + "/api/show", data=json.dumps({"model": "qwen3:1.7b"}).encode())
info = json.load(urllib.request.urlopen(req, timeout=30))["model_info"]  # chỉ đọc cấu hình, không sinh token

base = info["qwen3.rope.freq_base"]
head_dim = info["qwen3.attention.key_length"]
ctx = info["qwen3.context_length"]
print("architecture:", info["general.architecture"])
print("rope base:", base, "| head dim:", head_dim, "| context length:", ctx)

for b in (10000.0, base):
    wl = 2 * np.pi * b ** (np.arange(0, head_dim, 2) / head_dim)  # bước sóng của từng cặp chiều
    longest = wl[-1]
    slow = int((wl > ctx).sum())
    print(f"base {b:>9.0f}: fastest pair {wl[0]:.1f} tokens, slowest pair {longest:,.0f} tokens, pairs slower than context: {slow}/{head_dim // 2}")
