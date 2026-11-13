SYSTEM = {"role": "system", "content": "Bạn là trợ lý chat của Giao Hang Anh Ti ở Pho Rom Ra. Trả lời ngắn gọn bằng tiếng Việt."}
TURNS = [
    ("Chào bạn, tôi là Lan. Đơn của tôi là AT047838, giao tới Ben Song.", "Chào Lan, mình đã ghi nhận đơn AT047838 giao tới Ben Song."),
    ("Đơn này mấy giờ tới nơi?", "Đơn đang trên đường, dự kiến tới sau khoảng 25 phút."),
    ("Sao lâu vậy, tôi đặt từ 9 giờ sáng rồi.", "Xin lỗi Lan, khu vực Ben Song đang mưa nên shipper đi chậm hơn."),
    ("Shipper có gọi trước khi tới không?", "Có, shipper sẽ gọi trước khoảng 5 phút."),
    ("Tôi muốn đổi người nhận thành Minh được không?", "Được, bạn gửi giúp số điện thoại của Minh."),
    ("Số của Minh là 0901 234 567.", "Mình đã cập nhật người nhận là Minh, số 0901 234 567."),
    ("Có mất thêm phí không?", "Không, đổi người nhận không tính thêm phí."),
    ("Nếu Minh không nghe máy thì sao?", "Shipper sẽ gọi lại hai lần rồi liên hệ bạn."),
    ("Được rồi, cảm ơn.", "Không có gì, chúc Lan một ngày tốt lành."),
    ("À mà đơn đó giao tới quận nào nhỉ?", None),
]


def messages_until(n):
    msgs = [SYSTEM]
    for user, bot in TURNS[:n]:
        msgs.append({"role": "user", "content": user})
        if bot is not None:
            msgs.append({"role": "assistant", "content": bot})
    return msgs
