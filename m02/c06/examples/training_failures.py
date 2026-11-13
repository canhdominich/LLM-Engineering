import torch

from minigpt import GPT, CharTokenizer, GPTConfig, load_kieu

torch.set_num_threads(1)
text = load_kieu()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text))
train, val = data[:int(0.9 * len(data))], data[int(0.9 * len(data)):]


def batch(src, gen, size=32, block=64, shift=1):
    ix = torch.randint(len(src) - block - 1, (size,), generator=gen)
    # shift=0 là lỗi cố ý: y trùng x, quên dịch một ký tự
    return torch.stack([src[i:i + block] for i in ix]), torch.stack([src[i + shift:i + shift + block] for i in ix])


def run(name, lr, causal=True, shift=1, steps=200):
    torch.manual_seed(1337)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, causal=causal))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    gen, curve = torch.Generator().manual_seed(1), []
    for step in range(1, steps + 1):
        _, loss = model(*batch(train, gen, shift=shift))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            curve.append(f"{loss.item():.2f}")
    model.eval()
    with torch.no_grad():
        val_loss = model(*batch(val, torch.Generator().manual_seed(0), shift=shift))[1].item()
        sample = model.generate(torch.tensor([tok.encode("Trăm năm")]), 30, temperature=0)
    print(f"{name:<16} train loss at 50/100/150/200: {' '.join(curve)} | val {val_loss:.2f}")
    print(f"{'':<16} greedy sample: {tok.decode(sample[0].tolist())!r}")


run("baseline lr=3e-3", 3e-3)
run("lr=1e-1", 1e-1)
run("no causal mask", 3e-3, causal=False)
run("y not shifted", 3e-3, shift=0)
