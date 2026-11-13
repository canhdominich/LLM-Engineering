import math

from bigram_counts import count_bigrams, load_sentences

following = count_bigrams(load_sentences())
vocab = {w for c in following.values() for w in c} | set(following)
V = len(vocab)


def prob(prev: str, nxt: str, k: float) -> float:
    total = sum(following[prev].values())
    return (following[prev][nxt] + k) / (total + k * V)  # k = 0: đếm thuần; k = 1: add-one


def perplexity(sentence: str, k: float) -> float:
    words = ["<s>"] + sentence.split() + ["</s>"]
    log_sum = 0.0
    for prev, nxt in zip(words, words[1:]):
        p = prob(prev, nxt, k)
        if p == 0:
            return math.inf
        log_sum += math.log(p)
    return math.exp(-log_sum / (len(words) - 1))


print(f"vocabulary size V = {V}")
for s in ["đơn <order> đã giao thành công chưa", "shipper giao trễ quá", "giao trễ quá shipper"]:
    print(f"{s!r:38} count-only {perplexity(s, 0):>8.1f}   add-one {perplexity(s, 1):>7.1f}")
print(f"P('quá' | 'trễ'): count-only {prob('trễ', 'quá', 0):.3f}, add-one {prob('trễ', 'quá', 1):.4f}")
