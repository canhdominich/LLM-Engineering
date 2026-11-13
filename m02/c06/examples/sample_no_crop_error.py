import torch

from minigpt import load_checkpoint

model, tok = load_checkpoint()
idx = torch.tensor([tok.encode("Trăm năm")])
with torch.no_grad():
    for step in range(80):
        # lỗi cố ý: đưa cả chuỗi vào model, không cắt còn block_size token cuối
        logits, _ = model(idx)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
        if idx.shape[1] % 16 == 0:
            print("length", idx.shape[1])
