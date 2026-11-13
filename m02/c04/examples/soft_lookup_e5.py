import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from sentence_transformers import SentenceTransformer

KEYS = ["giao hàng trễ, shipper đến muộn so với giờ hẹn", "hàng bị vỡ, móp méo hoặc hư hỏng khi nhận",
        "đổi địa chỉ nhận hàng sang nơi khác", "hủy đơn hàng và yêu cầu hoàn tiền"]
HOURS = np.array([24.0, 48.0, 4.0, 72.0])
QUERIES = ["đơn của tôi tới muộn cả tiếng rồi", "hộp bánh bị bẹp hết rồi",
           "giao đến công ty thay vì nhà được không", "tôi không muốn nhận nữa, trả lại tiền giúp tôi"]

model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
K = model.encode(["passage: " + k for k in KEYS], normalize_embeddings=True)
Q = model.encode(["query: " + q for q in QUERIES], normalize_embeddings=True)
print("K shape:", K.shape, " Q shape:", Q.shape)

S = Q @ K.T  # cosine vì mọi vector đã có độ dài 1
print("cosine (rows = queries, cols = late, damaged, address, cancel):")
print(S.round(3))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


print("query 0 with different scales:")
for scale in (1, 10, 50, 100):
    w = softmax(S[0] * scale)
    print(f"  scale={scale:<3} weights={w.round(3)} answer={w @ HOURS:5.1f} hours")

W = softmax(S * 100)  # cả 4 câu hỏi trong một phép tính
print("scale=100 answers:", (W @ HOURS).round(1), " top key:", W.argmax(axis=1))
