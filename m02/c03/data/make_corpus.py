"""Simulated support messages of Giao Hang Anh Ti for training word embeddings.

Rules (seed 42, 6,000 messages, Vietnamese with diacritics):
- intents: late 30%, status 14%, cancel 13%, refund 11%, address 11%, goods 11%, early 7%, slow_rare (3 messages).
- "trễ", "chậm", "muộn" are interchangeable in late messages (weights 5:3:2).
- "sớm" appears in the same slots as the late words, but only in early messages.
- "hủy" and "huỷ" are the same word written with two tone placements (weights 3:1).
- "shipper" and "tài xế" are interchangeable (weights 3:2).
- "đường" means road in late messages (đường ngập, đường kẹt xe) and sugar in goods orders of Tạp Hóa Cô Tấm.
- "lề mề" (sluggish) appears in exactly 3 messages.
"""
import csv
import random
from pathlib import Path

N = 6000
DISTRICTS = ["Đầu Phố", "Cầu Gỗ", "Xóm Vui", "Bến Sông", "Bãi Biển", "Chợ Mới", "Đồi Thông", "Khu Công Nghiệp"]
GOODS = ["gạo", "sữa", "mì", "đường", "bánh", "nước mắm"]
ROAD = ["ngập", "kẹt xe", "đang sửa", "trơn"]

TEMPLATES = {
    "late": [
        "{ship} giao {late} quá, đơn {oid} vẫn chưa tới",
        "sao hôm nay giao {late} vậy",
        "đơn {oid} lại giao {late} {n} phút",
        "{ship} đến {late} hơn giờ hẹn {n} phút",
        "đường {road} nên {ship} giao {late} hả",
        "trời mưa đường {road}, giao {late} cũng được nhưng báo tôi nhé",
        "giao {late} thế này lần sau tôi {cancel} đơn",
        "{ship} báo sẽ đến {late} vì đường {road}",
    ],
    "early": [
        "{ship} đến {early} quá, tôi chưa về nhà",
        "giao {early} hơn giờ hẹn {n} phút, tôi không kịp ra nhận",
        "đơn {oid} giao {early} quá, nhờ {ship} gửi bảo vệ",
        "{ship} đến {early} hơn giờ hẹn, tôi chưa chuẩn bị tiền",
    ],
    "cancel": [
        "tôi muốn {cancel} đơn {oid}",
        "{cancel} giúp tôi đơn {oid} nhé",
        "làm sao để {cancel} đơn đã đặt",
        "đặt nhầm rồi, {cancel} đơn này giúp tôi",
        "{cancel} đơn {oid} thì có mất phí không",
    ],
    "refund": [
        "khi nào tôi được hoàn tiền đơn {oid}",
        "{cancel} đơn rồi mà chưa thấy hoàn tiền",
        "hoàn tiền giúp tôi vào ví nhé",
        "hàng bị vỡ, cho tôi xin hoàn tiền",
    ],
    "address": [
        "đổi địa chỉ giao sang {dist} giúp tôi",
        "địa chỉ mới của tôi ở {dist}",
        "{ship} giao nhầm địa chỉ ở {dist} rồi",
        "tôi chuyển nhà sang {dist}, sửa địa chỉ giúp tôi",
    ],
    "status": [
        "đơn {oid} đang ở đâu rồi",
        "bao giờ {ship} tới {dist}",
        "kiểm tra giúp tôi đơn {oid} với",
        "{ship} đã nhận đơn {oid} chưa",
    ],
    "goods": [
        "giao thêm {k} gói {good} cho Tạp Hóa Cô Tấm",
        "đơn của Tạp Hóa Cô Tấm thiếu một gói {good}",
        "{good} và {good2} giao chung một đơn nhé",
        "Tạp Hóa Cô Tấm đặt {k} kg {good} và {k2} thùng {good2}",
    ],
}
WEIGHTS = {"late": 30, "status": 14, "cancel": 13, "refund": 11, "address": 11, "goods": 11, "early": 7}
RARE = ["{ship} lề mề quá, đơn {oid} chưa tới", "sao {ship} lề mề vậy", "giao lề mề quá, tôi {cancel} đơn đây"]


def fill(rnd, template):
    goods = rnd.sample(GOODS, 2)
    return template.format(
        ship=rnd.choices(["shipper", "tài xế"], [3, 2])[0],
        late=rnd.choices(["trễ", "chậm", "muộn"], [5, 3, 2])[0],
        early="sớm",
        cancel=rnd.choices(["hủy", "huỷ"], [3, 1])[0],
        oid=f"AT{rnd.randint(1, 50000):06d}",
        n=rnd.choice([10, 15, 20, 30, 40, 45]),
        road=rnd.choice(ROAD),
        dist=rnd.choice(DISTRICTS),
        good=goods[0], good2=goods[1],
        k=rnd.randint(2, 9), k2=rnd.randint(2, 5),
    )


def main():
    rnd = random.Random(42)
    intents = list(WEIGHTS)
    rows = []
    for i in range(N - len(RARE)):
        intent = rnd.choices(intents, [WEIGHTS[x] for x in intents])[0]
        rows.append((intent, fill(rnd, rnd.choice(TEMPLATES[intent]))))
    rows += [("late", fill(rnd, t)) for t in RARE]
    rnd.shuffle(rows)
    out = Path(__file__).resolve().parent / "support_corpus.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["message_id", "text", "intent"])
        for i, (intent, text) in enumerate(rows):
            w.writerow([f"S{i + 1:05d}", text, intent])
    print(f"wrote {len(rows)} messages to {out.name}")


if __name__ == "__main__":
    main()
