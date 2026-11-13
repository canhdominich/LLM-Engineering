import re
import unicodedata

PROMISES = ["hoan tien", "hoan lai tien", "tra lai tien", "boi thuong", "den bu", "voucher", "ma giam gia", "mien phi don"]
ORDER_RE = re.compile(r"\bAT\d{6}\b", re.I)
MINUTES_RE = re.compile(r"(\d+)\s*phut")


def fold(text: str) -> str:
    s = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)


def has(folded: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", folded) is not None


def check_answer(answer: str, customer_id: str, owner_of, user_texts: list[str], minutes_from_tools: set[int]) -> str | None:
    """Return a violation code or None. owner_of(order_id) gives the owner customer id or None."""
    # TODO 19: trả "promise:<cụm>" nếu câu trả lời hứa bồi thường (dùng PROMISES và has),
    # "ungrounded_minutes:<n>" nếu có "<n> phút" không đến từ predict_eta hay từ chính khách gõ,
    # "foreign_order:<id>" nếu nhắc mã đơn của khách khác mà khách không tự gõ. Không vi phạm thì trả None.
    return None
