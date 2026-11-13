import torch
import torch.nn.functional as F

from _anhti_text import HANDBOOK
from _tinygpt import answer, cached, generate, setup

questions = ["phí giao nội quận bao nhiêu?", "giao nhanh mất bao lâu?", "phí giao hàng cồng kềnh bao nhiêu?"]


@torch.no_grad()
def handbook_loss(model, tok):
    ids = torch.tensor([tok.encode("\n".join(HANDBOOK[:8]))[:96]])
    return F.cross_entropy(model(ids[:, :-1])[0], ids[0, 1:]).item()


tok, fresh = setup()
print(f"TinyGPT: {sum(p.numel() for p in fresh.parameters()):,} params, vocab {tok.vocab_size} characters")
tok, base, _ = cached("pretrain")
print(f"loss on handbook text: random init {handbook_loss(fresh, tok):.2f}, pretrained {handbook_loss(base, tok):.2f}")
print("pretrained, prompt 'Phí giao ':", generate(base, tok, "Phí giao "))
for q in questions:
    print(f"pretrained | {q} | {answer(base, tok, q)}")

tok, chat, _ = cached("sft")
for q in questions:
    print(f"after SFT  | {q} | {answer(chat, tok, q)}")
