import json
import math
import os
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
req = urllib.request.Request(f"{HOST}/api/show", data=json.dumps({"model": "qwen3:1.7b", "verbose": True}).encode())
info = json.load(urllib.request.urlopen(req, timeout=60))
cfg = {k.split(".", 1)[1]: v for k, v in info["model_info"].items() if k.startswith("qwen3.")}
print({k: cfg[k] for k in ("block_count", "embedding_length", "attention.head_count",
                           "attention.head_count_kv", "feed_forward_length", "context_length")})
print("vocab size:", len(info["model_info"]["tokenizer.ggml.tokens"]), "| quantization:", info["details"]["quantization_level"])


def group(name):
    if name.startswith("token_embd"):
        return "token embedding"
    if name.startswith("output."):
        return "output (lm_head)"
    if "norm" in name:
        return "norms"
    return "attention" if ".attn_" in name else "MLP (ffn)"


totals = {}
for t in info["tensors"]:
    totals[group(t["name"])] = totals.get(group(t["name"]), 0) + math.prod(t["shape"])
all_params = sum(totals.values())
for name, n in sorted(totals.items(), key=lambda x: -x[1]):
    print(f"{name:<18} {n / 1e6:>8.1f} M {n / all_params:>6.1%}")
print(f"{'total':<18} {all_params / 1e6:>8.1f} M")
print("block 0 tensors:", [(t["name"][6:], t["shape"]) for t in info["tensors"] if t["name"].startswith("blk.0.") and "norm" not in t["name"]])
