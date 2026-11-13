import torch

from minigpt import GPT, CharTokenizer, GPTConfig, load_kieu

torch.manual_seed(1337)
torch.set_num_threads(1)
text = load_kieu()
tok = CharTokenizer(text)
lines = text.splitlines(keepends=True)
small = torch.tensor(tok.encode("".join(lines[:300])))
val = torch.tensor(tok.encode("".join(lines[2925:])))
print(f"train on verses 1-300 only: {len(small):,} chars; val: verses 2926-3254, {len(val):,} chars")


def batch(src, gen, size=32, block=64):
    ix = torch.randint(len(src) - block - 1, (size,), generator=gen)
    return torch.stack([src[i:i + block] for i in ix]), torch.stack([src[i + 1:i + 1 + block] for i in ix])


@torch.no_grad()
def loss_on(src):
    model.eval()
    gen = torch.Generator().manual_seed(0)
    value = sum(model(*batch(src, gen))[1].item() for _ in range(10)) / 10
    model.train()
    return value


model = GPT(GPTConfig(vocab_size=tok.vocab_size, dropout=0.0))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
gen = torch.Generator().manual_seed(1)
print(f"{'step':>5} {'train':>6} {'val':>6} {'gap':>6}")
for step in range(601):
    if step % 100 == 0:
        tr, va = loss_on(small), loss_on(val)
        print(f"{step:>5} {tr:>6.3f} {va:>6.3f} {va - tr:>6.3f}")
    if step == 600:
        break
    _, loss = model(*batch(small, gen))
    opt.zero_grad()
    loss.backward()
    opt.step()
