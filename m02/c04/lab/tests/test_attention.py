import numpy as np
import pytest
import torch
import torch.nn.functional as F

import attention as at

rng = np.random.default_rng(42)


def t(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float64))


def test_softmax_rows_sum_to_one_and_match_torch():
    x = rng.normal(size=(3, 5))
    p = at.softmax(x, axis=-1)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.allclose(p, torch.softmax(t(x), dim=-1).numpy())


def test_softmax_is_stable_for_large_scores():
    p = at.softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.isfinite(p).all()
    assert np.allclose(p, [0.09003057, 0.24472847, 0.66524096])


def test_softmax_fully_masked_row_gives_zeros_not_nan():
    p = at.softmax(np.array([[-np.inf, -np.inf], [0.0, -np.inf]]), axis=-1)
    assert not np.isnan(p).any()
    assert p.tolist() == [[0.0, 0.0], [1.0, 0.0]]


def test_causal_mask_is_lower_triangle_of_allowed_positions():
    m = at.causal_mask(4)
    assert m.dtype == bool and m.shape == (4, 4)
    assert np.array_equal(m, np.tril(np.ones((4, 4), dtype=bool)))


def test_sdpa_matches_torch_without_mask():
    q, k, v = rng.normal(size=(2, 3, 5, 8)), rng.normal(size=(2, 3, 5, 8)), rng.normal(size=(2, 3, 5, 4))
    ours = at.scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(t(q), t(k), t(v)).numpy()
    assert ours.shape == (2, 3, 5, 4)
    assert np.allclose(ours, ref)


def test_sdpa_matches_torch_causal():
    q, k, v = (rng.normal(size=(4, 6, 8)) for _ in range(3))
    ours = at.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(t(q), t(k), t(v), is_causal=True).numpy()
    assert np.allclose(ours, ref)


def test_sdpa_matches_torch_with_boolean_padding_mask():
    q, k, v = (rng.normal(size=(5, 8)) for _ in range(3))
    mask = np.array([True, True, True, False, False])[None, :].repeat(5, axis=0)
    ours = at.scaled_dot_product_attention(q, k, v, mask=mask)
    ref = F.scaled_dot_product_attention(t(q)[None], t(k)[None], t(v)[None], attn_mask=torch.from_numpy(mask)).numpy()[0]
    assert np.allclose(ours, ref)


def test_sdpa_accepts_different_number_of_keys():
    out = at.scaled_dot_product_attention(rng.normal(size=(2, 8)), rng.normal(size=(7, 8)), rng.normal(size=(7, 3)))
    assert out.shape == (2, 3)


def test_sdpa_rejects_mismatched_query_and_key_dims():
    with pytest.raises(ValueError):
        at.scaled_dot_product_attention(np.ones((3, 8)), np.ones((3, 4)), np.ones((3, 4)))


def test_split_and_merge_heads_round_trip():
    x = rng.normal(size=(5, 12))
    h = at.split_heads(x, 3)
    assert h.shape == (3, 5, 4)
    assert np.array_equal(h[1], x[:, 4:8])
    assert np.array_equal(at.merge_heads(h), x)
    with pytest.raises(ValueError):
        at.split_heads(x, 5)


def _weights(d):
    return [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4)]


@pytest.mark.parametrize("causal", [False, True])
def test_multi_head_matches_torch_multihead_attention(causal):
    n, d, heads = 6, 16, 4
    x = rng.normal(size=(n, d))
    Wq, Wk, Wv, Wo = _weights(d)
    ours = at.multi_head_attention(x, Wq, Wk, Wv, Wo, heads, causal=causal)
    mha = torch.nn.MultiheadAttention(d, heads, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        mha.in_proj_weight.copy_(t(np.concatenate([Wq.T, Wk.T, Wv.T])))
        mha.out_proj.weight.copy_(t(Wo.T))
        mask = torch.nn.Transformer.generate_square_subsequent_mask(n, dtype=torch.float64) if causal else None
        ref, _ = mha(t(x)[None], t(x)[None], t(x)[None], attn_mask=mask, need_weights=False)
    assert ours.shape == (n, d)
    assert np.allclose(ours, ref[0].numpy())


def test_future_leak_is_zero_only_with_causal_mask():
    x = rng.normal(size=(6, 8))
    Wq, Wk, Wv, Wo = _weights(8)
    assert at.future_leak(x, Wq, Wk, Wv, Wo, 2, causal=True) == pytest.approx(0.0, abs=1e-12)
    assert at.future_leak(x, Wq, Wk, Wv, Wo, 2, causal=False) > 1e-3
