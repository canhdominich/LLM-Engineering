import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

NAME = "intfloat/multilingual-e5-small"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME).eval()
short = "query: huỷ đơn giúp tôi"
long = "query: shipper giao trễ quá, đơn của Tạp Hóa Cô Tấm vẫn chưa tới, tôi muốn hủy và hoàn tiền"


def embed(sentences, use_mask):
    enc = tok(sentences, return_tensors="pt", padding=True)
    with torch.no_grad():
        h = model(**enc).last_hidden_state                # (batch, seq, 384)
    if use_mask:
        m = enc["attention_mask"].unsqueeze(-1).float()   # 1 cho token thật, 0 cho padding
        return (h * m).sum(1) / m.sum(1)
    return h.mean(1)                                      # lỗi: tính cả token padding


for use_mask in [False, True]:
    alone = embed([short], use_mask)[0]
    in_batch = embed([short, long], use_mask)[0]
    cos = torch.nn.functional.cosine_similarity(alone, in_batch, dim=0).item()
    print(f"use_mask={use_mask!s:5}  cosine(alone, in batch) = {cos:.4f}")

print("tokens:", len(tok(short)["input_ids"]), "vs", len(tok(long)["input_ids"]))
st = SentenceTransformer(NAME).encode([short, long], convert_to_tensor=True)
manual = torch.nn.functional.normalize(embed([short, long], use_mask=True), dim=1)
print("sentence-transformers equals masked mean + L2 normalize:", torch.allclose(st, manual, atol=1e-5))
