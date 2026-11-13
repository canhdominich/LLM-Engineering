import codecs

import tiktoken

text = "Đơn trễ hẹn, khiếu nại ngay"
for name in ("cl100k_base", "o200k_base"):
    enc = tiktoken.get_encoding(name)
    pieces = [enc.decode_single_token_bytes(i) for i in enc.encode(text)]
    shown, broken = [], 0
    for b in pieces:
        try:
            shown.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            broken += 1
            shown.append(b.hex(" "))  # mảnh không phải ký tự UTF-8 trọn vẹn
    print(f"{name}: {len(pieces)} tokens, {broken} not valid UTF-8 alone")
    print("  ", shown)
    naive = "".join(b.decode("utf-8", errors="replace") for b in pieces)
    decoder = codecs.getincrementaldecoder("utf-8")()  # giữ byte dở dang chờ token sau
    streamed = "".join(decoder.decode(b) for b in pieces)
    print("   decode each token:", naive)
    print("   incremental      :", streamed, "| equal to text:", streamed == text)
