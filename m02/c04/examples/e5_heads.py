import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

NAME = "intfloat/multilingual-e5-small"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME, attn_implementation="eager").eval()
enc = tok("query: Đơn AT027075 giao trễ hai ngày, tôi muốn hủy nó", return_tensors="pt")
with torch.no_grad():
    out = model(**enc, output_attentions=True, output_hidden_states=True)
tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
print(len(tokens), "tokens:", " ".join(tokens))

X = out.hidden_states[0][0].numpy()            # đầu vào của tầng 0, shape (20, 384)
sa = model.encoder.layer[0].attention.self
H, n = model.config.num_attention_heads, X.shape[0]
d_head = X.shape[1] // H


def heads(lin):
    M = X @ lin.weight.detach().numpy().T + lin.bias.detach().numpy()
    return M.reshape(n, H, d_head).transpose(1, 0, 2)  # (12, 20, 32)


Q, K = heads(sa.query), heads(sa.key)
S = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head)
A = np.exp(S - S.max(axis=-1, keepdims=True))
A = A / A.sum(axis=-1, keepdims=True)
print("ours vs model attentions, max abs diff:", float(np.abs(A - out.attentions[0][0].numpy()).max()))

i = np.arange(1, n)
print("head  to previous token  to <s>  entropy")
for h in range(H):
    entropy = -(A[h] * np.log(A[h])).sum(axis=-1).mean()
    print(f"{h:4d}  {A[h][i, i - 1].mean():17.2f}  {A[h][:, 0].mean():6.2f}  {entropy:7.2f}")
print(f"uniform weight 1/{n} = {1 / n:.2f}, max entropy ln({n}) = {np.log(n):.2f}")
