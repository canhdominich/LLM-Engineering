import torch
import torch.nn as nn

torch.manual_seed(0)
D, N_EXPERTS, TOP_K, N_TOKENS = 64, 8, 2, 4096
CAPACITY = int(1.25 * N_TOKENS * TOP_K / N_EXPERTS)


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(D, N_EXPERTS, bias=False)
        self.register_buffer("skew", torch.zeros(N_EXPERTS))
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)) for _ in range(N_EXPERTS))

    def forward(self, x):
        scores = (self.router(x) + self.skew).softmax(dim=-1)
        weights, chosen = scores.topk(TOP_K, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            # chỉ những token chọn expert e mới chạy qua expert e
            rows, slot = (chosen == e).nonzero(as_tuple=True)
            if len(rows):
                out[rows] += weights[rows, slot, None] * expert(x[rows])
        return out, chosen


def report(name, chosen):
    load = torch.bincount(chosen.flatten(), minlength=N_EXPERTS)
    over = (load - CAPACITY).clamp(min=0).sum().item()
    print(f"{name}: {load.tolist()} | max/avg {load.max().item() / load.float().mean().item():.2f} | over capacity {over}")


moe = MoE()
expert_params = sum(p.numel() for p in moe.experts[0].parameters())
total = sum(p.numel() for p in moe.parameters())
active = TOP_K * expert_params + moe.router.weight.numel()
print(f"total params {total:,} | active per token {active:,} ({active / total:.0%}) | capacity per expert {CAPACITY}")
x = torch.randn(N_TOKENS, D)
with torch.no_grad():
    out, chosen = moe(x)
    print("output shape", tuple(out.shape))
    report("balanced router", chosen)
    moe.skew[0] = 3.0  # router lệch: expert 0 được cộng điểm với mọi token
    report("skewed router  ", moe(x)[1])
