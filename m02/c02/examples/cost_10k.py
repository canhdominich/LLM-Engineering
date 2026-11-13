import tiktoken

PRICE_IN, PRICE_OUT = 0.40, 1.60  # USD cho 1 triệu token, số minh họa
OVERHEAD = 4  # token đánh dấu vai trò cho mỗi tin nhắn, giả định
N_CONVERSATIONS = 10_000

SYSTEM = {
    "vi": "Bạn là trợ lý chăm sóc khách hàng của Giao Hàng Anh Tí. Trả lời ngắn gọn, lịch sự. Nếu khách hỏi về đơn, hãy hỏi mã đơn nếu chưa có. Không hứa hoàn tiền khi chưa kiểm tra.",
    "en": "You are the customer support assistant of Giao Hang Anh Ti. Answer briefly and politely. If the customer asks about an order, ask for the order code if it is missing. Do not promise refunds before checking.",
}
TURNS = {
    "vi": [("Đơn AT027075 của tôi đang ở đâu vậy?", "Đơn AT027075 đang được shipper giao, dự kiến tới trong 25 phút nữa."),
           ("Tôi đi vắng lúc 17 giờ, đơn có tới trước giờ đó không?", "Có ạ, đơn dự kiến tới lúc 16 giờ 35, trước giờ anh đi vắng."),
           ("Cho mình đổi địa chỉ giao sang Chợ Mới được không?", "Đơn đã rời kho nên không đổi được địa chỉ. Anh có thể hẹn shipper giao lại ở Chợ Mới.")],
    "en": [("Where is my order AT027075 right now?", "Order AT027075 is out for delivery and should arrive in about 25 minutes."),
           ("I will be out at 5 pm. Will the order arrive before then?", "Yes, it is expected at 4:35 pm, before you leave."),
           ("Can you change the delivery address to Cho Moi?", "The order has left the warehouse, so the address cannot be changed. You can ask the shipper to redeliver at Cho Moi.")],
}


def one_conversation(enc, system, turns):
    n = lambda text: len(enc.encode(text)) + OVERHEAD
    history, per_turn, tokens_out = n(system), [], 0
    for user, reply in turns:
        history += n(user)
        per_turn.append(history)  # mỗi lượt gửi lại toàn bộ lịch sử
        tokens_out += n(reply)
        history += n(reply)
    return per_turn, tokens_out


for enc_name in ("o200k_base", "cl100k_base"):
    enc = tiktoken.get_encoding(enc_name)
    print(enc_name)
    for name, system, lang in [("all English", SYSTEM["en"], "en"), ("all Vietnamese", SYSTEM["vi"], "vi"),
                               ("en system + vi chat", SYSTEM["en"], "vi")]:
        per_turn, t_out = one_conversation(enc, system, TURNS[lang])
        t_in = sum(per_turn)
        cost = N_CONVERSATIONS * (t_in * PRICE_IN + t_out * PRICE_OUT) / 1_000_000
        print(f"  {name:19} in={t_in:3} {per_turn} out={t_out:3}  {N_CONVERSATIONS:,} conversations: {cost:4.2f} USD")
