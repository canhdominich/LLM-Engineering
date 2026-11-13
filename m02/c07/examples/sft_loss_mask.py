import torch
import torch.nn.functional as F

from _anhti_text import SFT_PAIRS
from _tinygpt import cached, sft_example, template

tok, chat, _ = cached("sft")
q, a = SFT_PAIRS[0]
x, labels = sft_example(tok, q, a, chat.cfg.block_size)
used = [t for t in labels if t != -100]
print(repr(template(q, a)))
print(f"real positions: {len(template(q, a)) }, padding to block_size {chat.cfg.block_size}: {chat.cfg.block_size - len(template(q, a))}")
print(f"counted in loss (answer + newline): {len(used)}: {tok.decode(used)!r}")

logits = chat(torch.tensor([x]))[0]
y = torch.tensor(labels)
masked = F.cross_entropy(logits, y, ignore_index=-100)
prompt_part = F.cross_entropy(logits[:len(tok.encode(template(q))) - 1], torch.tensor(tok.encode(template(q)))[1:])
print(f"loss on answer only: {masked.item():.3f}")
print(f"loss on the question part (not trained by SFT): {prompt_part.item():.3f}")
