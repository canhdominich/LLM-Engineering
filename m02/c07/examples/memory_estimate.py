GB = 1024 ** 3

MODELS = {
    "1.7B-like": dict(n=1.7e9, d=2048, f=6144, layers=28),
    "7B-like": dict(n=6.7e9, d=4096, f=11008, layers=32),
}


def lora_params(d, f, layers, r):
    # q, k, v, o là 4 ma trận d x d; gate, up, down là 3 ma trận d x f
    return layers * (4 * r * (d + d) + 3 * r * (d + f))


def estimate(n, lora):
    return {
        "full fine-tune": n * 16,
        "LoRA (bf16 base)": n * 2 + lora * 16,
        "QLoRA (4-bit base)": n * 4.127 / 8 + lora * 16,
    }


for name, m in MODELS.items():
    lora = lora_params(m["d"], m["f"], m["layers"], r=16)
    print(f"{name}: LoRA r=16 trains {lora / 1e6:.1f}M params ({lora / m['n']:.2%})")
    for method, bytes_ in estimate(m["n"], lora).items():
        print(f"  {method:<20} {bytes_ / GB:6.1f} GB")
