import math

import torch

from minigpt import GPT, CharTokenizer, GPTConfig, load_kieu

tok = CharTokenizer(load_kieu())
a = torch.tensor([tok.encode("Trăm năm trong cõi người ta,")])
b = torch.tensor([tok.encode("Trăm năm trong cõi người tA,")])
changed = 26
print("sequence length:", a.shape[1], "| changed position:", changed)

for causal in (True, False):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, causal=causal)).eval()
    with torch.no_grad():
        la, loss = model(a[:, :-1], a[:, 1:])
        lb, _ = model(b[:, :-1], b[:, 1:])
    diff = (la - lb).abs().amax(dim=-1)[0]
    print(f"causal={causal!s:<5} max change at positions 0..{changed - 1}: {diff[:changed].max().item():.2e}"
          f" | at position {changed}: {diff[changed].item():.2e}")
    print(f"             loss at init {loss.item():.3f} vs ln(vocab) {math.log(tok.vocab_size):.3f}")
