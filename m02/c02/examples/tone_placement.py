import re
import unicodedata

TONES = "̣̀́̃̉"  # huyền, sắc, ngã, hỏi, nặng
# oa, oe, uy ở cuối âm tiết: dời dấu thanh từ nguyên âm sau về nguyên âm trước
PATTERN = re.compile(rf"(?<![qQ])([oOuU])([aAeEyY])([{TONES}])(?![^\W\d_])")


def nfc(text):
    return unicodedata.normalize("NFC", text)


def unify_tone(text):
    nfd = unicodedata.normalize("NFD", text)
    return nfc(PATTERN.sub(r"\1\3\2", nfd))


pairs = [("hòa", "hoà"), ("thủy", "thuỷ"), ("khỏe", "khoẻ"), ("hỏa tốc", "hoả tốc")]
for a, b in pairs:
    print(f"{a!r:12} vs {b!r:12} NFC equal: {nfc(a) == nfc(b)!s:5}  after unify_tone: {unify_tone(a) == unify_tone(b)}")

for word in ["quá", "hoàng", "của", "Đơn hoả tốc"]:
    print(f"{word!r:16} becomes {unify_tone(word)!r}")
