import numpy as np

from _sgns import load_corpus, neighbors, train

vocab, index, ids, freq = load_corpus()
W, _ = train(ids, freq, dim=16, window=2, k=5, epochs=3, lr=0.025, seed=0)
unit = W / np.linalg.norm(W, axis=1, keepdims=True)
sims = unit @ unit[index["trễ"]]
rank = int((sims > sims[index["sớm"]]).sum())
print(f"cos(trễ, sớm) = {sims[index['sớm']]:.2f}, rank of 'sớm' among neighbors of 'trễ': {rank}")


def show(word):
    near = ", ".join(f"{w} {s:.2f}" for w, s in neighbors(W, vocab, index, word))
    print(f"{word:>6} (count {int(freq[index[word]]):4d}): {near}")


for word in ["sớm", "lề_mề", "đường"]:
    show(word)

kept, _, _, _ = load_corpus(min_count=5)
print("with min_count=5:", len(kept), "words, 'lề_mề' kept:", "lề_mề" in kept)
