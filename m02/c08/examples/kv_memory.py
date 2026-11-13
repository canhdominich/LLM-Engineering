import httpx

BASE = "http://localhost:11434/api"


def kv_bytes_per_token(model, bytes_per_value=2):
    info = httpx.post(f"{BASE}/show", json={"model": model}, timeout=60).json()["model_info"]
    arch = info["general.architecture"]
    layers = info[f"{arch}.block_count"]
    kv_heads = info[f"{arch}.attention.head_count_kv"]
    head_dim = info.get(f"{arch}.attention.key_length") or info[f"{arch}.embedding_length"] // info[f"{arch}.attention.head_count"]
    # 2 vì lưu cả key lẫn value, ở mọi lớp, cho mọi head key/value
    return 2 * layers * kv_heads * head_dim * bytes_per_value, (layers, kv_heads, head_dim), info[f"{arch}.context_length"]


for model in ("qwen3:1.7b", "smollm2:135m-instruct-q8_0"):
    per_tok, shape, max_ctx = kv_bytes_per_token(model)
    print(f"{model}: layers, kv_heads, head_dim = {shape}, {per_tok / 1024:.1f} KiB per token (fp16)")
    for ctx, users in ((4096, 1), (max_ctx, 1), (4096, 8)):
        print(f"  ctx {ctx:>5} x {users} user(s): {per_tok * ctx * users / 2**30:6.2f} GiB")

small = "smollm2:135m-instruct-q8_0"
sizes = {}
for ctx in (2048, 8192):
    httpx.post(f"{BASE}/generate", json={"model": small, "prompt": "hi", "stream": False, "keep_alive": "20s",
                                         "options": {"num_ctx": ctx, "num_predict": 1}}, timeout=300)
    loaded = httpx.get(f"{BASE}/ps", timeout=60).json()["models"]
    sizes[ctx] = next(m["size"] for m in loaded if m["name"] == small)
    print(f"{small} loaded with num_ctx {ctx}: {sizes[ctx] / 2**20:.1f} MiB in memory")
per_tok = kv_bytes_per_token(small)[0]
print(f"measured growth {(sizes[8192] - sizes[2048]) / 2**20:.1f} MiB, formula KV for 6144 tokens {per_tok * 6144 / 2**20:.1f} MiB")
