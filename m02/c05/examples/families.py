import torch
from transformers import AutoConfig, AutoModel

MODELS = {  # tên model: (trường số lớp encoder, trường số lớp decoder)
    "google-bert/bert-base-uncased": ("num_hidden_layers", None),
    "openai-community/gpt2": (None, "n_layer"),
    "google-t5/t5-small": ("num_layers", "num_decoder_layers"),
}

print(f"{'model':<18} {'params':>12} {'enc':>4} {'dec':>4} {'self-attn':>9} {'cross-attn':>10}")
for name, (enc_key, dec_key) in MODELS.items():
    cfg = AutoConfig.from_pretrained(name)  # chỉ tải config.json
    with torch.device("meta"):  # dựng khung model, không cấp bộ nhớ cho trọng số
        model = AutoModel.from_config(cfg)
    params = sum(p.numel() for p in model.parameters())
    mods = [n for n, _ in model.named_modules()]
    enc = getattr(cfg, enc_key) if enc_key else 0
    dec = getattr(cfg, dec_key) if dec_key else 0
    selfa = sum(n.endswith((".attention.self", ".attn", ".SelfAttention")) for n in mods)
    cross = sum(n.endswith(".EncDecAttention") for n in mods)  # decoder đọc encoder
    print(f"{name.split('/')[1]:<18} {params:>12,} {enc:>4} {dec:>4} {selfa:>9} {cross:>10}")
