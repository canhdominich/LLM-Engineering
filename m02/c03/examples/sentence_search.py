from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")
FAQ = [
    "Khi trời mưa hoặc đường ngập, đơn có thể giao trễ 30 đến 60 phút.",
    "Đơn tạp hóa thiếu hàng như gạo, đường, sữa sẽ được giao bù miễn phí.",
    "Bạn có thể hủy đơn miễn phí trước khi shipper nhận hàng.",
    "Tiền được hoàn về ví trong 3 ngày làm việc.",
    "Đổi địa chỉ giao được một lần khi đơn chưa rời kho.",
]
QUESTIONS = ["đường ngập quá, shipper có giao không", "tạp hóa hết đường rồi, giao thêm nhé",
             "huỷ đơn giúp tôi", "giao lề mề quá"]

docs = model.encode([f"passage: {x}" for x in FAQ], normalize_embeddings=True)
queries = model.encode([f"query: {q}" for q in QUESTIONS], normalize_embeddings=True)
print("embedding shape:", docs.shape)
scores = queries @ docs.T
for q, row in zip(QUESTIONS, scores):
    best = row.argmax()
    print(f"{q:40} FAQ {best + 1}  score {row[best]:.3f}  runner-up {sorted(row)[-2]:.3f}")
