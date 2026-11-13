import unicodedata
from collections import Counter

MARK = "▁"  # đánh dấu đầu từ, giống SentencePiece


def word_counts(texts, form="NFC"):
    counts = Counter()
    for text in texts:
        for w in unicodedata.normalize(form, text).lower().split():
            counts[tuple(MARK + w)] += 1
    return counts


def pair_counts(words):
    pairs = Counter()
    for symbols, n in words.items():
        for pair in zip(symbols, symbols[1:]):
            pairs[pair] += n
    return pairs


def merge(symbols, pair):
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(texts, num_merges, form="NFC"):
    words = word_counts(texts, form)
    merges = []
    for _ in range(num_merges):
        pairs = pair_counts(words)
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))  # nhiều nhất; hòa thì theo thứ tự chữ
        merges.append((best, pairs[best]))
        words = Counter({merge(s, best): n for s, n in words.items()})
    return merges, words


if __name__ == "__main__":
    corpus = ["giao hàng gấp", "hàng giao trễ", "đơn hàng giao trễ", "giao gấp", "đơn giao trễ"]
    start = word_counts(corpus)
    print("words:", {"".join(s): n for s, n in start.items()})
    print("start symbols:", len({c for s in start for c in s}), " symbols in corpus:", sum(len(s) * n for s, n in start.items()))
    merges, words = train_bpe(corpus, 10)
    for step, ((a, b), n) in enumerate(merges, 1):
        print(f"merge {step}: {a!r} + {b!r} = {a + b!r}  (seen {n} times)")
    print("final:", {" ".join(s): n for s, n in words.items()}, " symbols in corpus:", sum(len(s) * n for s, n in words.items()))

    nfd = word_counts(corpus, "NFD")
    print("NFD start symbols:", len({c for s in nfd for c in s}), " symbols in corpus:", sum(len(s) * n for s, n in nfd.items()))
    nfd_merges, words = train_bpe(corpus, 10, "NFD")
    show = lambda t: "".join(f"<{ord(c):04X}>" if unicodedata.combining(c) else c for c in t)
    print("NFD merges:", [show(a + b) for (a, b), _ in nfd_merges])
    print("NFD after 10 merges, symbols in corpus:", sum(len(s) * n for s, n in words.items()))
