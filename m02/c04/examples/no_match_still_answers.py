import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from sentence_transformers import SentenceTransformer

KEYS = ["giao hàng trễ, shipper đến muộn so với giờ hẹn", "hàng bị vỡ, móp méo hoặc hư hỏng khi nhận",
        "đổi địa chỉ nhận hàng sang nơi khác", "hủy đơn hàng và yêu cầu hoàn tiền"]
HOURS = np.array([24.0, 48.0, 4.0, 72.0])
QUERIES = ["bên bạn có bán thùng carton không", "shipper làm móp hộp bánh của tôi"]

model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
K = model.encode(["passage: " + k for k in KEYS], normalize_embeddings=True)
Q = model.encode(["query: " + q for q in QUERIES], normalize_embeddings=True)

S = Q @ K.T
W = np.exp(S * 100 - (S * 100).max(axis=1, keepdims=True))
W = W / W.sum(axis=1, keepdims=True)
for q, s, w in zip(QUERIES, S, W):
    print(q)
    print(f"  cosine={s.round(3)} max={s.max():.3f}")
    print(f"  weights={w.round(3)} sum={w.sum():.3f} answer={w @ HOURS:.1f} hours")
