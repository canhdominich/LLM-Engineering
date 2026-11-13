import math
import unicodedata

import numpy as np
import pytest

import skipgram as sg


def test_tokenize_lowercases_and_joins_compounds():
    assert sg.tokenize("Tài xế giao TRỄ quá, đơn AT012345 chưa tới") == ["tài_xế", "giao", "trễ", "quá", "đơn", "chưa", "tới"]


def test_tokenize_same_result_for_nfc_and_nfd():
    text = "hủy đơn ở Xóm Vui"
    assert sg.tokenize(unicodedata.normalize("NFD", text)) == sg.tokenize(text) == ["hủy", "đơn", "ở", "xóm_vui"]


def test_build_vocab_sorted_by_count_then_word():
    vocab, counts = sg.build_vocab([["b", "a", "b"], ["c", "a", "b"]])
    assert vocab == ["b", "a", "c"]
    assert counts.tolist() == [3.0, 2.0, 1.0]
    assert sg.build_vocab([["b", "a", "b"], ["c", "a", "b"]], min_count=2)[0] == ["b", "a"]


def test_make_pairs_window_one():
    assert sg.make_pairs([[0, 1, 2]], window=1).tolist() == [[0, 1], [1, 0], [1, 2], [2, 1]]


def test_make_pairs_do_not_cross_messages():
    pairs = sg.make_pairs([[0, 1, 2, 3], [4, 5]], window=2)
    assert pairs.shape == (12, 2)
    assert [3, 4] not in pairs.tolist() and [4, 3] not in pairs.tolist()


def test_noise_distribution_uses_power():
    p = sg.noise_distribution([16.0, 1.0], power=0.75)
    assert math.isclose(p.sum(), 1.0)
    assert math.isclose(p[0] / p[1], 8.0)


def test_sgns_loss_is_k_plus_one_ln2_when_output_vectors_are_zero():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(3, 4))
    losses, *_ = sg.sgns_grads(v, np.zeros((3, 4)), np.zeros((3, 5, 4)))
    assert losses.shape == (3,)
    assert np.allclose(losses, 6 * math.log(2))


def test_sgns_grads_match_finite_differences():
    rng = np.random.default_rng(2)
    v, u_pos, u_neg = rng.normal(size=(1, 4)), rng.normal(size=(1, 4)), rng.normal(size=(1, 3, 4))
    _, grad_v, grad_pos, grad_neg = sg.sgns_grads(v, u_pos, u_neg)
    eps = 1e-6
    for arr, grad in [(v, grad_v), (u_pos, grad_pos), (u_neg, grad_neg)]:
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + eps
            up = sg.sgns_grads(v, u_pos, u_neg)[0].sum()
            arr[idx] = old - eps
            down = sg.sgns_grads(v, u_pos, u_neg)[0].sum()
            arr[idx] = old
            num[idx] = (up - down) / (2 * eps)
        assert np.allclose(grad, num, atol=1e-5)


def test_most_similar_excludes_word_and_sorts():
    vocab = ["trễ", "chậm", "gạo", "sớm"]
    W = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.7, 0.7]])
    result = sg.most_similar(W, vocab, "trễ", k=3)
    assert [w for w, _ in result] == ["chậm", "sớm", "gạo"]
    assert result[0][1] > result[1][1] > result[2][1]
    with pytest.raises(KeyError):
        sg.most_similar(W, vocab, "lề_mề")


def test_synonym_hit_rate_counts_top1_in_group():
    vocab = ["trễ", "chậm", "gạo", "sữa"]
    W = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.8, 0.6]])
    assert math.isclose(sg.synonym_hit_rate(W, vocab, [["trễ", "chậm"], ["gạo", "sữa"], ["không_có"]]), 0.75)


def test_rare_word_removed_by_min_count():
    vocab, _ = sg.build_vocab(sg.load_docs(), min_count=5)
    assert "lề_mề" not in vocab and "trễ" in vocab


def test_training_lowers_loss_and_finds_synonyms():
    docs = sg.load_docs()
    vocab, counts = sg.build_vocab(docs)
    index = {w: i for i, w in enumerate(vocab)}
    W, history = sg.train([[index[w] for w in d] for d in docs], counts, epochs=2)
    assert history[-1] < 0.5 * 6 * math.log(2)
    top2 = {w for w, _ in sg.most_similar(W, vocab, "trễ", k=2)}
    assert top2 == {"chậm", "muộn"}
    assert sg.synonym_hit_rate(W, vocab, sg.GROUPS) >= 0.9
