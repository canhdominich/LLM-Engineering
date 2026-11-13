import json
import os
import urllib.request

from transformers import AutoConfig

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
req = urllib.request.Request(HOST + "/api/show", data=json.dumps({"model": "qwen3:1.7b"}).encode())
info = json.load(urllib.request.urlopen(req, timeout=30))["model_info"]
hf = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")  # chỉ tải config.json, không tải trọng số

d = info["qwen3.embedding_length"]
ff = info["qwen3.feed_forward_length"]
n_layers = info["qwen3.block_count"]
h, h_kv, hd = info["qwen3.attention.head_count"], info["qwen3.attention.head_count_kv"], info["qwen3.attention.key_length"]
print(f"d_model={d} d_ff={ff} layers={n_layers} heads={h} kv_heads={h_kv} head_dim={hd} vocab={hf.vocab_size}")

attn = d * h * hd + 2 * d * h_kv * hd + h * hd * d + 2 * hd  # Q, K, V, O và chuẩn hóa q, k
mlp = 3 * d * ff  # SwiGLU: gate, up, down
norms = 2 * d
block = attn + mlp + norms
emb = hf.vocab_size * d
print(f"per block: attention {attn:,} + mlp {mlp:,} + norms {norms:,} = {block:,}")
print(f"{n_layers} blocks: {n_layers * block:,}")
print(f"embedding table: {emb:,} | tied output: {hf.tie_word_embeddings}")
print(f"total (tied): {n_layers * block + emb + d:,}")
print(f"total if output stored separately: {n_layers * block + 2 * emb + d:,}")
print(f"ollama reports: {info['general.parameter_count']:,}")
