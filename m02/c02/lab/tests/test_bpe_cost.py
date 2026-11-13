import unicodedata
from collections import Counter

import pytest

import bpe_cost as bc

TOY = ["giao hàng gấp", "hàng giao trễ", "đơn hàng giao trễ", "giao gấp", "đơn giao trễ"]


def test_normalize_nfc_spaces_and_case():
    raw = unicodedata.normalize("NFD", "Khu  Công Nghiệp ")
    assert bc.normalize(raw) == "khu công nghiệp"
    assert bc.normalize(raw) == unicodedata.normalize("NFC", bc.normalize(raw))


def test_pair_counts_weighted_by_word_frequency():
    words = Counter({("▁", "a", "b"): 3, ("▁", "a", "c"): 1})
    pairs = bc.pair_counts(words)
    assert pairs[("▁", "a")] == 4
    assert pairs[("a", "b")] == 3
    assert pairs[("a", "c")] == 1


def test_merge_replaces_every_non_overlapping_occurrence():
    assert bc.merge(("a", "b", "a", "b"), ("a", "b")) == ("ab", "ab")
    assert bc.merge(("a", "a", "a"), ("a", "a")) == ("aa", "a")
    assert bc.merge(("x", "y"), ("a", "b")) == ("x", "y")


def test_train_bpe_first_merges_on_toy_corpus():
    merges = bc.train_bpe(TOY, 4)
    assert merges == [("▁", "g"), ("a", "o"), ("i", "ao"), ("▁g", "iao")]


def test_train_bpe_stops_when_nothing_left_to_merge():
    merges = bc.train_bpe(["ab ab"], 50)
    assert merges == [("a", "b"), ("▁", "ab")]


def test_encode_applies_merges_by_rank_not_left_to_right():
    merges = [("b", "c"), ("a", "b")]
    assert bc.encode("abc", merges) == ["▁", "a", "bc"]


def test_encode_known_and_unknown_words():
    merges = bc.train_bpe(TOY, 10)
    assert bc.encode("giao hàng", merges) == ["▁giao", "▁hàng"]
    assert bc.encode("giaoo", merges) == ["▁giao", "o"]


def test_encode_is_the_same_for_nfc_and_nfd_and_decodes_back():
    merges = bc.train_bpe([p[0] for p in bc.PAIRS], 60)
    text = "Đơn hỏa tốc trễ hẹn ở Khu Công Nghiệp"
    nfd = unicodedata.normalize("NFD", text)
    assert bc.encode(nfd, merges) == bc.encode(text, merges)
    assert bc.decode(bc.encode(text, merges)) == bc.normalize(text)


def test_count_tokens_with_tiktoken():
    assert bc.count_tokens(["Where is my order?"], "o200k_base") == 5
    assert bc.count_tokens(["Đơn hàng của tôi đang ở đâu?"], "o200k_base") == 9
    assert bc.count_tokens(["Đơn hàng của tôi đang ở đâu?", "Where is my order?"], "cl100k_base") == 20


def test_language_ratio_vietnamese_costs_more_on_older_tokenizer():
    new = bc.language_ratio(bc.PAIRS, "o200k_base")
    old = bc.language_ratio(bc.PAIRS, "cl100k_base")
    assert 1.0 < new < old
    assert old > 1.5


def test_conversation_tokens_resends_history():
    count = lambda text: len(text.split())
    turns = [("a b", "c"), ("d", "e f g")]
    assert bc.conversation_tokens("s s s", turns, count, overhead=0) == (3 + 2 + 3 + 2 + 1 + 1, 1 + 3)
    assert bc.conversation_tokens("s s s", turns, count, overhead=1) == ((4 + 3) + (4 + 3 + 2 + 2), 2 + 4)


def test_estimate_cost():
    assert bc.estimate_cost(1000, 200, 10_000, 0.40, 1.60) == pytest.approx(7.2)
    assert bc.estimate_cost(0, 0, 10_000, 0.40, 1.60) == 0
