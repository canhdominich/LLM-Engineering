import csv
import random
import unicodedata
from pathlib import Path

DISTRICTS = {
    "Dau Pho": "Đầu Phố", "Cau Go": "Cầu Gỗ", "Xom Vui": "Xóm Vui", "Ben Song": "Bến Sông",
    "Bai Bien": "Bãi Biển", "Cho Moi": "Chợ Mới", "Doi Thong": "Đồi Thông", "Khu Cong Nghiep": "Khu Công Nghiệp",
}
PLACES = ["Tạp Hóa Cô Tấm", "Cà Phê Chú Tư", "Quán Cơm Bác Tám", "Tiệm Bánh Cô Mận", "Nhà Sách Ông Đồ",
          "Trà Sữa Cô Na", "Phở Bà Năm", "Hoa Nhà Mận", "Giặt Là Cô Út"]
UNITS = ["thùng", "gói", "hộp", "túi", "bó hoa"]
WORDS = {1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm"}

TEMPLATES = [
    ("std_or_exp", "Lấy hàng ở {place} quận {pu}, giao {n} {unit} tới {do}, thu hộ {cod}{tail}"),
    ("std_or_exp", "Nhờ shop qua {place} bên {pu} lấy {n} {unit} giao sang {do}{tail}. Người nhận trả {cod}"),
    ("exp_prepaid", "cần giao gấp {n} {unit} từ {pu} đi {do}, khách trả trước rồi"),
    ("std", "Mình ở {pu}, muốn gửi {n} {unit} cho bạn ở {do}, thu hộ giúp mình {cod}"),
    ("std", "Đặt đơn: lấy tại {place}, {pu}; giao {do}; {n} {unit}; COD {cod}; giao thường"),
    ("std", "{do} nhận {n} {unit} từ {place} ở {pu} nha, thu {cod}"),
    ("exp_prepaid", "giao {n} {unit} từ {pu} qua {do} trong 1 tiếng, không thu tiền"),
    ("exp", "Hỏa tốc: {place} ({pu}) gửi {n} {unit} đến {do}, thu hộ {cod}"),
    ("std", "Chuyển {n} {unit} từ {pu} về {do} giúp tôi, tiền hàng {cod} người nhận trả, không cần nhanh"),
    ("exp_prepaid", "bên mình ở {pu} có {n} {unit} cần đưa tới {do}, giao nhanh giúp, đã chuyển khoản"),
]


def strip_marks(text):
    s = unicodedata.normalize("NFD", text)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D").lower()


def cod_text(rnd, k):
    forms = [f"{k}k", f"{k:,}.000đ".replace(",", "."), f"{k} nghìn", f"{k * 1000} đồng"]
    if k >= 1000 and k % 100 == 0:
        m, r = divmod(k, 1000)
        forms += [f"{m}tr{r // 100}" if r else f"{m}tr", f"{k / 1000:g} triệu".replace(".", ",")]
    return rnd.choice(forms)


def make(n_rows=40, seed=42):
    rnd = random.Random(seed)
    rows = []
    for i in range(n_rows):
        kind, tpl = TEMPLATES[i % len(TEMPLATES)]
        pu, do = rnd.sample(list(DISTRICTS), 2)
        n = rnd.randint(1, 12)
        k = rnd.choice([rnd.randrange(50, 1000, 10), rnd.choice([1000, 1200, 1500, 2000, 2500])])
        service = "express" if kind.startswith("exp") or (kind == "std_or_exp" and rnd.random() < 0.4) else "standard"
        cod = 0 if kind == "exp_prepaid" else k * 1000
        tail = rnd.choice([", giao gấp nha", ", hỏa tốc"]) if (kind == "std_or_exp" and service == "express") else ""
        n_text = WORDS[n] if n in WORDS and rnd.random() < 0.35 else str(n)
        text = tpl.format(place=rnd.choice(PLACES), pu=DISTRICTS[pu], do=DISTRICTS[do], n=n_text,
                          unit=rnd.choice(UNITS), cod=cod_text(rnd, k), tail=tail)
        if rnd.random() < 0.3:
            text = strip_marks(text)
        rows.append({"request_id": f"R{i + 1:02d}", "text": text, "pickup_district": pu, "dropoff_district": do,
                     "items": n, "cod_vnd": cod, "service": service})
    return rows


if __name__ == "__main__":
    rows = make()
    out = Path(__file__).resolve().parent / "delivery_requests.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out.name, len(rows), "rows")
