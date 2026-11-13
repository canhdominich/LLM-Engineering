import torch
import torch.nn.functional as F

from block_forward import Block


class PlainBlock(Block):
    def forward(self, x):
        x = self.attn(self.ln1(x))  # bỏ đường tắt: chỉ còn đầu ra của nhánh
        x = self.ff(self.ln2(x))
        return x


def probe(cls, depth):
    torch.manual_seed(0)
    blocks = torch.nn.Sequential(*[cls() for _ in range(depth)])
    x = torch.randn(4, 6, 64, requires_grad=True)  # 4 tin nhắn, 6 token
    y = blocks(x)
    F.mse_loss(y, torch.randn(4, 6, 64)).backward()
    u = F.normalize(y, dim=-1)
    sim = u @ u.transpose(1, 2)
    same = (sim.sum() - sim.diagonal(dim1=1, dim2=2).sum()).item() / (4 * 6 * 5)  # cosine giữa hai token khác nhau
    keep = F.cosine_similarity(y, x, dim=-1).mean().item()  # đầu ra còn giống đầu vào của chính token đó
    return same, keep, x.grad.norm().item()


print("depth | token-token cos  | keeps own input | grad at input")
print("      |  res     plain   |  res     plain  |  res     plain")
for depth in [1, 2, 6, 24]:
    r, p = probe(Block, depth), probe(PlainBlock, depth)
    print(f"{depth:>5} | {r[0]:.3f}   {p[0]:.3f}   | {r[1]:.3f}   {p[1]:>6.3f}  | {r[2]:.4f}  {p[2]:.4f}")
