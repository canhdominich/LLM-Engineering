import torch
import torch.nn as nn

torch.manual_seed(0)


class LoRALinear(nn.Module):
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.base.bias.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / r)
        # B bằng 0 nên lúc bắt đầu lớp mới cho kết quả y hệt lớp gốc
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged(self):
        layer = nn.Linear(self.base.in_features, self.base.out_features)
        layer.weight.data = self.base.weight + self.scale * self.B @ self.A
        layer.bias.data = self.base.bias.clone()
        return layer


base = nn.Linear(64, 192)
lora = LoRALinear(base, r=4, alpha=8)
x = torch.randn(5, 64)

full = sum(p.numel() for p in base.parameters())
train = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(f"trainable params in this LoRALinear (r=4): {train}")
print(f"full layer params: {full}")
for r in (1, 4, 16, 64):
    n = r * (64 + 192)
    print(f"LoRA r={r:<2} trains {n:5d} params ({n / full:.1%} of the layer)")
print(f"same idea on a 4096 x 4096 layer, r=16: {16 * 8192 / 4096 ** 2:.2%}")
print(f"max diff at step 0: {(lora(x) - base(x)).abs().max().item():.1f}")

w0 = base.weight.clone()
target = base(x) + 0.5 * torch.randn(5, 192)
opt = torch.optim.SGD([lora.A, lora.B], lr=0.05)
print(f"loss at step 0: {((lora(x) - target) ** 2).mean().item():.4f}")
for step in range(200):
    loss = ((lora(x) - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"loss after 200 steps: {loss.item():.4f}")
print(f"base weight unchanged: {torch.equal(base.weight, w0)}")
print(f"rank of B @ A: {torch.linalg.matrix_rank(lora.B @ lora.A).item()}")
merged = lora.merged()
print(f"max diff merged vs adapter: {(merged(x) - lora(x)).abs().max().item():.1e}")
