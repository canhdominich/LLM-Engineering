import torch

from minigpt import filter_probs, load_checkpoint

model, tok = load_checkpoint()


def show(ch):
    return {" ": "space", "\n": "newline"}.get(ch, ch)


for prompt in ("Trăm năm trong cõi ngườ", "Trăm năm trong cõi người ta,\n"):
    with torch.no_grad():
        logits = model(torch.tensor([tok.encode(prompt)]))[0][:, -1, :]
    print(f"prompt {prompt[-12:]!r}")
    for t in (0.5, 1.0, 1.5):
        probs = torch.softmax(logits / t, dim=-1)[0]
        top = probs.topk(4)
        cells = " ".join(f"{show(tok.chars[i])}={p:.2f}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"  T={t}: {cells}")
    for name, kw in (("top_k=5", dict(top_k=5)), ("top_p=0.9", dict(top_p=0.9))):
        kept = int((filter_probs(logits, **kw) > 0).sum())
        print(f"  {name}: keeps {kept} of {tok.vocab_size} chars")
