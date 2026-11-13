from transformers import AutoConfig, AutoTokenizer

cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
d, layers = cfg.hidden_size, cfg.num_hidden_layers
kv = cfg.num_key_value_heads * cfg.head_dim
per_layer = 2 * d * d + 2 * d * kv + 3 * d * cfg.intermediate_size  # attention + MLP, bỏ qua norm
embed = cfg.vocab_size * d
total = embed + layers * per_layer
print(f"Qwen3-1.7B: vocab={cfg.vocab_size} hidden={d} layers={layers} tied embeddings={cfg.tie_word_embeddings}")
print(f"embedding table: {embed / 1e6:.0f}M params = {embed / total:.0%} of about {total / 1e9:.2f}B")

for vocab in (256, 32_000, cfg.vocab_size, 1_000_000):
    print(f"  vocab {vocab:>9,}: embedding {vocab * d / 1e6:7.1f}M params, {vocab * d * 2 / 2**20:6.0f} MiB in bf16")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
msg = "Đơn AT027075 của tôi đang ở đâu vậy?"
for name, n in [("characters", len(msg)), ("bytes", len(msg.encode("utf-8"))), ("qwen tokens", len(tok.encode(msg)))]:
    print(f"{name:12} n={n:3}  attention pairs n*n={n * n:5}")
