import tiktoken
from transformers import AutoTokenizer

o200k = tiktoken.get_encoding("o200k_base")
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def o200k_pieces(text):
    return [o200k.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in o200k.encode(text)]


def qwen_pieces(text):
    return [qwen.decode([i]) for i in qwen.encode(text)]


print(f"vocab size  o200k={o200k.n_vocab}  qwen={len(qwen)}")
for text in ["giao hàng", "phường", "phuong", "khongg", "AT047838"]:
    print(f"{text!r}")
    print("   o200k:", o200k_pieces(text))
    print("   qwen :", qwen_pieces(text))
