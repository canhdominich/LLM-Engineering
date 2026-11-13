import csv
import re
import unicodedata
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "support_corpus.csv"
COMPOUNDS = ["tài xế", "hoàn tiền", "địa chỉ", "giờ hẹn", "kẹt xe", "nước mắm", "bảo vệ", "lề mề",
             "tạp hóa cô tấm", "đầu phố", "cầu gỗ", "xóm vui", "bến sông", "bãi biển", "chợ mới",
             "đồi thông", "khu công nghiệp"]


def tokenize(text):
    # NFC trước để "ủ" gõ kiểu tổ hợp và kiểu dựng sẵn thành cùng một chuỗi
    text = unicodedata.normalize("NFC", text).lower()
    for phrase in COMPOUNDS:
        text = text.replace(phrase, phrase.replace(" ", "_"))
    return re.findall(r"\b[^\W\d]+\b", text)


def load_messages():
    with DATA.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
