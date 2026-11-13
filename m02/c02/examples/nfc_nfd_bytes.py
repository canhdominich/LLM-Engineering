import unicodedata

import tiktoken

district = "Khu Công Nghiệp"
nfc = unicodedata.normalize("NFC", district)
nfd = unicodedata.normalize("NFD", district)

print("look the same:", nfc, "|", nfd)
print("equal:", nfc == nfd)
print(f"code points  NFC={len(nfc)}  NFD={len(nfd)}")
print(f"UTF-8 bytes  NFC={len(nfc.encode('utf-8'))}  NFD={len(nfd.encode('utf-8'))}")

# Soi riêng chữ "ệ" ở hai dạng
for form in ("NFC", "NFD"):
    ch = unicodedata.normalize(form, "ệ")
    parts = [f"U+{ord(c):04X} {unicodedata.name(c)}" for c in ch]
    print(form, "ệ =", " + ".join(parts), "| bytes:", ch.encode("utf-8").hex(" "))

enc = tiktoken.get_encoding("o200k_base")
print(f"o200k tokens NFC={len(enc.encode(nfc))}  NFD={len(enc.encode(nfd))}")
print("NFC(NFD) == NFC:", unicodedata.normalize("NFC", nfd) == nfc)
