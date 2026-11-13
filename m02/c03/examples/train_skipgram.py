import time

from _sgns import load_corpus, neighbors, train

vocab, index, ids, freq = load_corpus()
start = time.perf_counter()
W, history = train(ids, freq, dim=16, window=2, k=5, epochs=3, lr=0.025, seed=0)
print(f"trained {W.shape} in {time.perf_counter() - start:.0f} s")
print("loss per epoch:", [round(x, 3) for x in history], "| loss at start: 6 x ln 2 = 4.159")
for word in ["trễ", "hủy", "shipper", "gạo", "xóm_vui"]:
    print(f"{word:>8}:", ", ".join(f"{w} {s:.2f}" for w, s in neighbors(W, vocab, index, word)))
