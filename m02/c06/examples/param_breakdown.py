import torch

from minigpt import GPT, CharTokenizer, GPTConfig, load_kieu

tok = CharTokenizer(load_kieu())
model = GPT(GPTConfig(vocab_size=tok.vocab_size))


def count(module):
    return sum(p.numel() for p in module.parameters())


groups = {
    "token embedding (132 x 64)": count(model.tok_emb),
    "position embedding (64 x 64)": count(model.pos_emb),
    "attention, 4 blocks": sum(count(b.attn) for b in model.blocks),
    "MLP, 4 blocks": sum(count(b.mlp) for b in model.blocks),
    "LayerNorm (8 in blocks + ln_f)": sum(count(b.ln1) + count(b.ln2) for b in model.blocks) + count(model.ln_f),
}
total = count(model)
for name, n in groups.items():
    print(f"{name:<32} {n:>8,} {n / total:>6.1%}")
print(f"{'total (lm_head shares tok_emb)':<32} {total:>8,}")
print("lm_head is tok_emb:", model.lm_head.weight is model.tok_emb.weight)
block = model.blocks[0]
print("one block:", count(block), "= attention", count(block.attn), "+ MLP", count(block.mlp), "+ LayerNorm", count(block.ln1) + count(block.ln2))
