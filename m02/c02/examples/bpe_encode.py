import unicodedata

from bpe_train import MARK, merge, train_bpe

corpus = ["giao hàng gấp", "hàng giao trễ", "đơn hàng giao trễ", "giao gấp", "đơn giao trễ"]
merges, _ = train_bpe(corpus, 10)
ranks = {pair: rank for rank, (pair, _) in enumerate(merges)}


def encode_word(word, ranks):
    symbols = tuple(MARK + word)
    while len(symbols) > 1:
        pairs = set(zip(symbols, symbols[1:]))
        best = min(pairs, key=lambda p: ranks.get(p, float("inf")))  # cặp học sớm nhất
        if best not in ranks:
            break
        symbols = merge(symbols, best)
    return list(symbols)


def encode(text, ranks):
    text = unicodedata.normalize("NFC", text).lower()
    return [t for w in text.split() for t in encode_word(w, ranks)]


def decode(tokens):
    return "".join(tokens).replace(MARK, " ").strip()


if __name__ == "__main__":
    for text in ["giao hàng", "hàng giao gấp", "giao trễ hẹn", "giaoo"]:
        tokens = encode(text, ranks)
        print(f"{text!r:16} {len(tokens):2} tokens {tokens}  decode ok: {decode(tokens) == text}")
