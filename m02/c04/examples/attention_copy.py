import torch

from _copy_task import evaluate, train

model, loss = train(hidden=32, attention=True)
params = sum(p.numel() for p in model.parameters())
print(f"with attention, hidden = 32, params = {params:,}, last loss = {loss:.3f}")
print("  length  digit accuracy  whole code right")
for n in (5, 10, 15, 20):
    digit, whole = evaluate(model, n)
    print(f"  {n:6d}  {digit:14.3f}  {whole:16.3f}")

phone = torch.tensor([[0, 9, 1, 2, 3, 4, 5, 6, 7, 8]])  # số điện thoại 10 chữ số
pred, maps = model.greedy(phone, torch.tensor([10]), 10)
print("input :", "".join(map(str, phone[0].tolist())))
print("output:", "".join(map(str, pred[0].tolist())))
print("step  looks most at input position  weight")
for t in (0, 1, 2, 5, 9):
    w = maps[t][0, 0]
    print(f"{t:4d}  {int(w.argmax()):30d}  {float(w.max()):6.2f}")
