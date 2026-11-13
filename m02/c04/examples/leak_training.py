import math

import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)
torch.set_num_threads(1)
n, d = 16, 32


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(10, d)
        self.pos = nn.Embedding(n, d)          # vị trí học được, Chương 5 sẽ nói kỹ
        self.qkv = nn.Linear(d, 3 * d)
        self.head = nn.Linear(d, 10)

    def forward(self, x, causal):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        return self.head(h + F.scaled_dot_product_attention(q, k, v, is_causal=causal))


def run(causal):
    model = TinyLM()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(1)
    for _ in range(300):
        x = torch.randint(0, 10, (64, n + 1), generator=gen)  # chữ số ngẫu nhiên, không đoán được
        logits = model(x[:, :-1], causal)
        loss = F.cross_entropy(logits.reshape(-1, 10), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    test = torch.randint(0, 10, (500, n + 1), generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        prefix = test[:, :8]  # lúc sinh thật: chỉ có 8 chữ số đã viết, chưa có tương lai
        gen_loss = F.cross_entropy(model(prefix, causal)[:, -1], test[:, 8]).item()
    return loss.item(), gen_loss


print(f"best possible loss on random digits = ln(10) = {math.log(10):.3f}")
for causal in (False, True):
    train_loss, gen_loss = run(causal)
    print(f"causal={causal!s:5s} train loss {train_loss:.3f}  loss when generating {gen_loss:.3f}")
