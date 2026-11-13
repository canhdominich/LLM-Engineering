import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
vocab = {i: tok.decode([i]) for i in range(len(tok))}
DISTRICTS = ["Dau Pho", "Cau Go", "Xom Vui", "Ben Song", "Bai Bien", "Cho Moi", "Doi Thong", "Khu Cong Nghiep"]
TARGETS = [d + '"' for d in DISTRICTS]  # giá trị enum kèm dấu nháy đóng


def allowed(done):
    # token hợp lệ nếu phần đã sinh cộng token vẫn là tiền tố của ít nhất một giá trị
    keep = []
    for i, piece in vocab.items():
        text = done + piece
        if piece and any(t.startswith(text) for t in TARGETS):
            keep.append(piece)
    return keep


print("vocab size:", len(vocab))
for done in ["", "B", "Bai", "Cho", "Cho Moi"]:
    t0 = time.perf_counter()
    keep = allowed(done)
    ms = (time.perf_counter() - t0) * 1000
    sample = sorted(keep, key=lambda p: (-len(p), p))[:6]
    print(f"after {done!r:10} allowed {len(keep):3d} ({ms:.0f} ms): {sample}")
print("tokens of value:", [tok.decode([i]) for i in tok.encode('Khu Cong Nghiep"')])
