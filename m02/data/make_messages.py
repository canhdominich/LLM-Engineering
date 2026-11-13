"""Generate the simulated Giao Hang Anh Ti support messages used across module 02."""
import random
import unicodedata
from pathlib import Path

import pandas as pd

SEED = 42
HERE = Path(__file__).resolve().parent

DISTRICTS = ["Đầu Phố", "Xóm Vui", "Chợ Mới", "Bến Sông", "Đồi Thông", "Cầu Gỗ", "Bãi Biển", "Khu Công Nghiệp"]
PLACES = ["Chung Cư Xóm Vui", "Cà Phê Chú Tư", "Tạp Hóa Cô Tấm", "Quán Cơm Bác Tám", "Công Viên Đầu Phố", "Nhà Sách Ông Đồ"]

TEMPLATES = {
    "order_status": [
        "Đơn {oid} của tôi đang ở đâu vậy?",
        "Cho mình hỏi tình trạng đơn {oid}",
        "Đơn hàng {oid} đã lấy hàng chưa shop?",
        "Kiểm tra giúp em đơn {oid} với ạ",
        "Đơn {oid} hiện đang ở trạng thái gì rồi",
        "Shipper đã nhận đơn {oid} chưa?",
        "Mình muốn tra cứu đơn {oid}",
        "Đơn của em sao vẫn báo đang xử lý từ hôm qua",
        "Đơn {oid} đã rời kho chưa ạ",
        "Đơn tôi đặt hôm qua giờ đang ở bước nào rồi",
        "Cho hỏi đơn {oid} đã giao thành công chưa vậy",
        "Trạng thái đơn {oid} là gì thế",
        "Đơn {oid} có bị hủy không, sao không thấy cập nhật gì",
        "App báo đơn {oid} giao thất bại là sao ạ",
        "Hàng của mình đã tới kho {d} chưa",
        "Mã vận đơn {oid} đang nằm ở đâu",
        "Đơn {oid} đã được shipper lấy chưa hay vẫn nằm ở shop",
        "Em không thấy đơn {oid} trong lịch sử, kiểm tra giúp em",
        "Đơn gửi từ {p} hôm nay đã được nhận chưa",
        "Sao đơn {oid} cứ báo đang trung chuyển mãi vậy",
        "Xem giúp tôi đơn {oid} đang do ai giao",
        "Đơn {oid} đã thanh toán chưa, app hiện chờ xác nhận",
        "Bên mình đã nhận đơn hàng tôi tạo lúc sáng chưa",
        "Đơn {oid} báo đã giao nhưng tôi chưa nhận được gì",
        "Cho em xin cập nhật mới nhất của đơn {oid}",
    ],
    "eta": [
        "Khi nào đơn {oid} tới nơi?",
        "Mấy giờ shipper tới {d} vậy?",
        "Đơn {oid} dự kiến giao lúc mấy giờ",
        "Bao lâu nữa thì hàng tới vậy shop",
        "Hôm nay có giao kịp trước {h} giờ không ạ",
        "Chiều nay em có nhận được đơn {oid} không",
        "Giao từ {p} tới {d} mất bao lâu",
        "Đơn {oid} khoảng mấy phút nữa thì tới",
        "Shipper còn cách nhà tôi bao xa",
        "Tôi đi vắng lúc {h} giờ, đơn có tới trước giờ đó không",
        "Ước tính giờ giao đơn {oid} giúp mình",
        "Đơn hỏa tốc đi {d} thì bao lâu nhận được",
        "Mai mấy giờ thì giao đơn {oid} ạ",
        "Có kịp giao trong buổi trưa nay không",
        "Đơn {oid} hôm nay có giao không hay để mai",
        "Còn bao lâu thì shipper tới {p}",
        "Mình cần hàng trước {h} giờ, đơn {oid} có kịp không",
        "Thời gian giao dự kiến của đơn {oid} là bao nhiêu",
        "Tầm mấy giờ thì đơn tới khu {d} vậy shop",
        "Đơn tiêu chuẩn đi {d} thường mất mấy tiếng",
        "Bao giờ shipper bắt đầu giao đơn {oid}",
        "Shipper sắp tới chưa, tôi chuẩn bị ra cổng",
    ],
    "complaint_late": [
        "Đơn {oid} trễ {n} phút rồi mà vẫn chưa thấy đâu",
        "Hẹn giao trong 1 tiếng mà giờ đã gần 2 tiếng",
        "Sao giao chậm thế, tôi chờ từ sáng tới giờ",
        "Lần thứ ba giao trễ rồi đấy, làm ăn kiểu gì vậy",
        "Đơn hỏa tốc mà trễ {n} phút, thất vọng quá",
        "Đã quá giờ hẹn {n} phút, shipper không gọi cũng không tới",
        "Tôi chờ đơn {oid} cả buổi chiều, quá chậm",
        "Giao trễ làm tôi lỡ cuộc họp, rất bực",
        "Chậm quá, đơn {oid} cam kết 60 phút mà giờ chưa tới",
        "Ngày nào cũng trễ, khách của tôi phàn nàn liên tục",
        "Trời không mưa mà vẫn giao trễ {n} phút là sao",
        "Đơn {oid} để quá giờ hẹn, đồ ăn nguội hết rồi",
        "Báo giao trước {h} giờ mà giờ đã quá {h} giờ rồi",
        "Mình rất không hài lòng vì đơn {oid} trễ hẹn",
        "Giao hàng chậm như rùa, chờ mãi không thấy",
        "Shipper đi đâu mất, trễ {n} phút rồi",
        "Hàng cần gấp mà giao trễ thế này thì dùng làm gì",
        "Đơn {oid} trễ hơn một tiếng, tôi muốn khiếu nại",
        "Khu {d} lần nào cũng bị giao chậm",
        "Tôi đã đợi ở {p} {n} phút mà shipper chưa đến",
    ],
    "refund": [
        "Tôi muốn hoàn tiền đơn {oid}",
        "Đơn bị hủy thì bao giờ tôi nhận lại tiền",
        "Hàng vỡ rồi, cho tôi xin lại tiền",
        "Đã trả trước mà đơn {oid} không giao, hoàn tiền giúp tôi",
        "Phí ship bị trừ hai lần, xin hoàn lại một lần",
        "Đơn {oid} trễ quá, hủy đi và trả lại tiền cho tôi",
        "Làm sao để được hoàn phí giao hàng",
        "Tiền hoàn của đơn {oid} bao lâu thì về tài khoản",
        "Mình đã hủy đơn mà chưa thấy tiền trả lại",
        "Giao sai hàng, tôi muốn lấy lại tiền",
        "Yêu cầu hoàn tiền vì đơn {oid} bị thất lạc",
        "Shop ơi trả lại tiền cọc đơn {oid} giúp em",
        "Đơn hỏa tốc trễ thì có được hoàn phí không",
        "Tôi bị tính phí sai, đề nghị hoàn lại phần chênh",
        "Hàng bị ướt do mưa, tôi muốn được bồi hoàn",
        "Hoàn tiền về ví hay về ngân hàng vậy",
        "Đơn {oid} đã hoàn tiền chưa, đợi một tuần rồi",
        "Thanh toán thừa {n} nghìn, trả lại giúp mình",
        "Đơn bị giao thất bại, xin hoàn lại phí ship",
        "Cho tôi hỏi thủ tục hoàn tiền khi hàng hỏng",
    ],
    "change_address": [
        "Cho mình đổi địa chỉ giao đơn {oid} sang {d}",
        "Em ghi nhầm số nhà, sửa giúp em với",
        "Giao tới {p} thay vì địa chỉ cũ được không",
        "Đổi người nhận và địa chỉ của đơn {oid}",
        "Tôi chuyển nhà sang {d}, cập nhật địa chỉ đơn {oid}",
        "Sửa địa chỉ nhận hàng thành {p} giúp tôi",
        "Đơn {oid} giao về công ty thay vì nhà được không",
        "Mình nhập sai phường, đổi giúp sang {d}",
        "Có đổi được địa chỉ khi shipper đã lấy hàng không",
        "Cho em đổi chỗ nhận sang {p} nhé",
        "Thay đổi địa chỉ giao đơn {oid}, tôi không có nhà hôm nay",
        "Nhờ shipper giao qua nhà hàng xóm số {n} được không",
        "Địa chỉ đơn {oid} bị thiếu số tầng, bổ sung giúp mình",
        "Muốn chuyển đơn {oid} sang khu {d}",
        "Cập nhật lại địa chỉ: giao tới {p}",
        "Tôi viết sai tên đường, sửa lại địa chỉ đơn {oid}",
        "Giao đơn {oid} ra {p} giúp em, em đang ở đó",
        "Đổi địa chỉ nhận đơn sang {d} có mất thêm phí không",
    ],
    "other": [
        "Shop có giao hàng ở {d} không",
        "Phí ship đi {d} là bao nhiêu",
        "Bên mình có tuyển shipper không ạ",
        "Cảm ơn shop nhé, giao nhanh lắm",
        "Có nhận giao hàng cồng kềnh không",
        "Làm sao để tạo đơn mới trên app",
        "Ứng dụng bị lỗi, không đăng nhập được",
        "Tổng đài làm việc tới mấy giờ",
        "Có gói giao hàng theo tháng cho cửa hàng không",
        "Muốn hợp tác giao hàng cho {p} thì liên hệ ai",
        "Shipper hôm nay rất dễ thương, khen một câu",
        "Có giao hàng vào chủ nhật không",
        "Mã giảm giá hôm nay còn dùng được không",
        "Cho hỏi cân nặng tối đa một đơn là bao nhiêu",
        "Làm sao đổi mật khẩu tài khoản",
        "Có nhận gửi đồ dễ vỡ không shop",
        "Shop có bán thùng carton không",
        "Xuất hóa đơn công ty thì làm thế nào",
        "Tôi muốn đăng ký làm tài xế xe tải",
        "Bên mình có giao thuốc không ạ",
        "Có thu hộ tiền hàng không, phí bao nhiêu",
        "Văn phòng ở {d} nằm chỗ nào vậy",
    ],
}

COUNTS = {"order_status": 60, "eta": 55, "complaint_late": 50, "refund": 45, "change_address": 45, "other": 45}

ABBREV = [("không", "ko"), ("được", "dc"), ("gì", "j"), ("vậy", "v"), ("rồi", "r"), ("bao giờ", "bh"), ("nhé", "nha"), ("với", "vs")]


def strip_accents(text):
    s = unicodedata.normalize("NFD", text)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def add_typo(text, rnd):
    for full, short in ABBREV:
        if full in text and rnd.random() < 0.7:
            return text.replace(full, short, 1)
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not idx:
        return text
    i = rnd.choice(idx)
    w = words[i]
    k = rnd.randrange(1, len(w) - 1)
    words[i] = w[:k] + w[k + 1:] if rnd.random() < 0.5 else w[:k - 1] + w[k] + w[k - 1] + w[k + 1:]
    return " ".join(words)


def fill(tpl, rnd):
    return tpl.format(oid=f"AT{rnd.randrange(0, 50000):06d}", d=rnd.choice(DISTRICTS), p=rnd.choice(PLACES),
                      h=rnd.choice([9, 10, 11, 14, 16, 17, 18]), n=rnd.choice([20, 25, 30, 40, 45, 50, 90]))


def make(rnd):
    rows, seen = [], set()
    for intent, n in COUNTS.items():
        tpls = TEMPLATES[intent]
        made = 0
        while made < n:
            tpl = tpls[made % len(tpls)] if made < len(tpls) else rnd.choice(tpls)
            text = fill(tpl, rnd)
            lower = rnd.random() < 0.45
            plain = rnd.random() < 0.4
            typo = rnd.random() < 0.2
            if lower:
                text = text[0].lower() + text[1:]
            if typo:
                text = add_typo(text, rnd)
            if plain:
                text = strip_accents(text).lower()
            if rnd.random() < 0.3:
                text = text.rstrip("?.!")
            key = strip_accents(text).lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "intent": intent, "has_diacritics": not plain, "has_typo": typo, "template_id": f"{intent}_{tpls.index(tpl):02d}"})
            made += 1
    rnd.shuffle(rows)
    nfd = set(rnd.sample([i for i, r in enumerate(rows) if r["has_diacritics"]], 12))
    for i, r in enumerate(rows):
        r["message_id"] = f"M{i + 1:04d}"
        r["unicode_form"] = "NFD" if i in nfd else "NFC"
        r["text"] = unicodedata.normalize(r["unicode_form"], r["text"])
    df = pd.DataFrame(rows)[["message_id", "text", "intent", "template_id", "has_diacritics", "has_typo", "unicode_form"]]
    # tách theo mẫu câu: mẫu nằm ở test thì không có câu nào cùng mẫu ở train
    test_templates = set()
    for intent in COUNTS:
        names = [f"{intent}_{i:02d}" for i in range(len(TEMPLATES[intent]))]
        test_templates |= set(rnd.sample(names, round(len(names) * 0.2)))
    df["split"] = ["test" if t in test_templates else "train" for t in df["template_id"]]
    return df


if __name__ == "__main__":
    df = make(random.Random(SEED))
    df.to_csv(HERE / "messages.csv", index=False, encoding="utf-8")
    print(df.shape, df["split"].value_counts().to_dict())
    print(df["intent"].value_counts().to_dict())
    print("plain:", (~df["has_diacritics"]).sum(), "typo:", df["has_typo"].sum(), "nfd:", (df["unicode_form"] == "NFD").sum())
