import csv
import json
import re
import unicodedata
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
DISTRICTS = {"Dau Pho", "Cau Go", "Xom Vui", "Ben Song", "Bai Bien", "Cho Moi", "Doi Thong", "Khu Cong Nghiep"}
KEYS = {"pickup_district", "dropoff_district", "items", "cod_vnd", "service"}
replies = json.loads((DATA / "free_text_replies.json").read_text(encoding="utf-8"))
gold = {r["request_id"]: r for r in csv.DictReader(open(DATA / "delivery_requests.csv", encoding="utf-8"))}


def loads_or_none(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


FENCE = "`" * 3  # ba dấu backtick của khung code markdown


def strip_fence(text):
    return re.sub(rf"^{FENCE}(?:json)?\s*|\s*{FENCE}$", "", text.strip())


def ascii_title(text):
    s = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").title()


def coerce(obj):
    # vá tay: lấy số đầu tiên trong chuỗi, bỏ dấu và viết hoa chữ đầu tên quận
    num = lambda v: int(re.search(r"\d+", str(v)).group())
    return {**obj, "items": num(obj["items"]), "cod_vnd": num(obj["cod_vnd"]),
            "pickup_district": ascii_title(obj["pickup_district"]), "dropoff_district": ascii_title(obj["dropoff_district"])}


def problems(obj):
    out = []
    if set(obj) != KEYS:
        out.append("keys")
    if not (isinstance(obj["items"], int) and isinstance(obj["cod_vnd"], int)):
        out.append("type")
    for k in ("pickup_district", "dropoff_district"):
        if obj[k] not in DISTRICTS:
            out.append(f"{k[:-9]}={obj[k]!r}")
    if obj["service"] not in ("express", "standard"):
        out.append("service")
    return out


print("raw json.loads ok:", sum(loads_or_none(t) is not None for t in replies.values()), "/ 6")
print("after strip_fence:", sum(loads_or_none(strip_fence(t)) is not None for t in replies.values()), "/ 6")
for rid, text in replies.items():
    obj = loads_or_none(strip_fence(text))
    fixed = coerce(obj)
    left = problems(fixed)
    wrong = [k for k in ("pickup_district", "dropoff_district", "items", "cod_vnd") if str(fixed[k]) != gold[rid][k]]
    print(f"{rid} before: {problems(obj)}")
    print(f"    after coerce: {left or 'valid'}; wrong vs gold: {wrong or 'none'}")
