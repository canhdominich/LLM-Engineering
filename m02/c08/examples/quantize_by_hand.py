import torch

torch.manual_seed(0)
W = torch.randn(1536, 384) * 0.04  # cỡ một lớp MLP của model embedding nhỏ


def quantize(w, bits, group=None):
    qmax = 2 ** (bits - 1) - 1  # int8: 127, int4: 7
    g = w.reshape(-1, group) if group else w.reshape(1, -1)
    scale = g.abs().amax(dim=1, keepdim=True) / qmax  # mỗi nhóm một thang đo
    q = torch.clamp(torch.round(g / scale), -qmax, qmax)  # số nguyên được lưu
    return (q * scale).reshape(w.shape), scale.numel()


def report(name, w, bits, group=None):
    w_hat, n_scales = quantize(w, bits, group)
    size_kb = (w.numel() * bits + n_scales * 16) / 8 / 1024
    err = (w - w_hat).abs()
    zeros = (w_hat == 0).float().mean() * 100  # trọng số bị làm tròn thành 0
    print(f"{name:<20}{size_kb:>8.1f} KB  mean err {err.mean():.5f}  max err {err.max():.5f}  zeros {zeros:4.1f}%")


print(f"{'fp32 original':<20}{W.numel() * 4 / 1024:>8.1f} KB")
report("int8 per-tensor", W, 8)
report("int4 per-tensor", W, 4)
report("int4 group 32", W, 4, 32)
W_out = W.clone()
W_out[0, 0] = 2.0  # một trọng số ngoại lai lớn gấp 50 lần độ lệch chuẩn
print("with one outlier = 2.0:")
report("  int4 per-tensor", W_out, 4)
report("  int4 group 32", W_out, 4, 32)
