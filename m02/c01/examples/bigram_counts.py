import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"


def load_sentences() -> list[list[str]]:
    df = pd.read_csv(DATA).query("split == 'train' and has_diacritics and unicode_form == 'NFC'")
    sentences = []
    for text in df["text"]:
        text = re.sub(r"\bat\d{6}\b", "<order>", text.lower())  # mọi mã đơn thành một từ chung
        sentences.append(["<s>"] + re.findall(r"<order>|\w+", text) + ["</s>"])
    return sentences


def count_bigrams(sentences: list[list[str]]) -> dict[str, Counter]:
    following = defaultdict(Counter)
    for words in sentences:
        for prev, nxt in zip(words, words[1:]):
            following[prev][nxt] += 1  # đếm: sau từ prev, từ nxt xuất hiện bao nhiêu lần
    return following


def next_word_probs(following: dict[str, Counter], prev: str) -> dict[str, float]:
    total = sum(following[prev].values())
    return {w: c / total for w, c in following[prev].most_common()}


if __name__ == "__main__":
    sentences = load_sentences()
    following = count_bigrams(sentences)
    print(f"sentences: {len(sentences)}, words: {sum(len(s) - 2 for s in sentences)}")
    print(f"distinct bigrams: {sum(len(c) for c in following.values())}")
    for prev in ["giao", "hoàn", "<s>"]:
        probs = next_word_probs(following, prev)
        top = ", ".join(f"{w} {p:.2f}" for w, p in list(probs.items())[:5])
        print(f"after {prev!r} ({sum(following[prev].values())} times): {top}")
