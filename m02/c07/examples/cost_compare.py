import tiktoken

from _policy import POLICY

enc = tiktoken.get_encoding("o200k_base")
SYSTEM = ("Bạn là trợ lý chăm sóc khách hàng của Giao Hàng Anh Tí. Trả lời ngắn, lịch sự, "
          "xưng mình và gọi khách là bạn. Nếu khách hỏi về đơn mà chưa có mã đơn thì xin mã đơn. "
          "Không hứa điều không có trong chính sách.")
QUESTION = "Phí giao nội quận của Giao Hàng Anh Tí hiện nay là bao nhiêu?"
CONTEXT = "\n".join(POLICY[:3])

variants = {
    "prompt only": SYSTEM + QUESTION,
    "prompt + RAG (3 chunks)": SYSTEM + CONTEXT + QUESTION,
    "fine-tuned, short prompt": QUESTION,
}
# giá giả định, chỉ để so sánh tương đối, không phải bảng giá thật
PRICE_IN = 0.40 / 1e6
REQUESTS = 10_000 * 4
for name, text in variants.items():
    n = len(enc.encode(text))
    print(f"{name:<26} {n:4d} tokens  {n * REQUESTS / 1e6:5.2f}M tokens/month  ${n * REQUESTS * PRICE_IN:6.2f}")

train_tokens = 2_000 * 60 * 3
print(f"one-time fine-tune: 2,000 examples x 60 tokens x 3 epochs = {train_tokens:,} training tokens")
