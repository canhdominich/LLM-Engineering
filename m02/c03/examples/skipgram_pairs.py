from _sgns import load_corpus, make_pairs
from _text import tokenize

message = "shipper giao trễ quá, đơn AT012345 vẫn chưa tới"
tokens = tokenize(message)
print("tokens:", tokens)
for window in [1, 2]:
    pairs = [(tokens[i], tokens[j]) for i in range(len(tokens))
             for j in range(max(0, i - window), min(len(tokens), i + window + 1)) if j != i]
    print(f"window={window}: {len(pairs)} pairs")
print("pairs with center 'trễ' (window=2):", [p for p in pairs if p[0] == "trễ"])

vocab, index, ids, freq = load_corpus()
all_pairs = make_pairs(ids, window=2)
print(f"whole corpus: {len(all_pairs)} pairs from {int(freq.sum())} tokens, vocabulary {len(vocab)}")
late, give = index["trễ"], index["giao"]
print("pair (trễ, giao) appears", int(((all_pairs[:, 0] == late) & (all_pairs[:, 1] == give)).sum()), "times")
noise = freq ** 0.75 / (freq ** 0.75).sum()
for w in ["đơn", "trễ", "lề_mề"]:
    print(f"P_noise({w}) = {noise[index[w]]:.4f}  vs  raw share {freq[index[w]] / freq.sum():.4f}")
