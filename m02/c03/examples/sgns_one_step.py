import numpy as np
import torch

rng = np.random.default_rng(0)
d = 4
v = rng.normal(0, 0.5, d)            # vector vào của "trễ"
u_pos = rng.normal(0, 0.5, d)        # vector ra của hàng xóm thật "giao"
u_neg = rng.normal(0, 0.5, (2, d))   # vector ra của 2 từ nhiễu "gạo", "địa_chỉ"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss_of(v, u_pos, u_neg):
    return -np.log(sigmoid(u_pos @ v)) - np.log(sigmoid(-(u_neg @ v))).sum()


s_pos, s_neg = sigmoid(u_pos @ v), sigmoid(u_neg @ v)
print(f"P(giao is a real neighbor) = {s_pos:.3f}")
print("P(noise words are neighbors) =", np.round(s_neg, 3))
print(f"loss before = {loss_of(v, u_pos, u_neg):.4f}")

grad_v = (s_pos - 1) * u_pos + (s_neg[:, None] * u_neg).sum(0)
grad_pos = (s_pos - 1) * v
grad_neg = s_neg[:, None] * v

tv, tp, tn = (torch.tensor(x, requires_grad=True) for x in (v, u_pos, u_neg))
t_loss = -torch.nn.functional.logsigmoid(tp @ tv) - torch.nn.functional.logsigmoid(-(tn @ tv)).sum()
t_loss.backward()
print("max |manual - autograd| =", f"{max(abs(grad_v - tv.grad.numpy()).max(), abs(grad_pos - tp.grad.numpy()).max(), abs(grad_neg - tn.grad.numpy()).max()):.1e}")

lr = 0.5
v2, u_pos2, u_neg2 = v - lr * grad_v, u_pos - lr * grad_pos, u_neg - lr * grad_neg
print(f"after one step: P(giao) = {sigmoid(u_pos2 @ v2):.3f}, P(noise) = {np.round(sigmoid(u_neg2 @ v2), 3)}")
print(f"loss after = {loss_of(v2, u_pos2, u_neg2):.4f}")
