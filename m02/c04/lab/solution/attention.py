import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)
    total = e.sum(axis=axis, keepdims=True)
    return np.divide(e, total, out=np.zeros_like(e), where=total > 0)


def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))


def scaled_dot_product_attention(q, k, v, mask=None, is_causal=False):
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (q, k, v))
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"query dim {q.shape[-1]} != key dim {k.shape[-1]}")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(f"{k.shape[-2]} keys but {v.shape[-2]} values")
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    allowed = np.ones(scores.shape[-2:], dtype=bool)
    if is_causal:
        allowed = allowed & causal_mask(scores.shape[-1])[: scores.shape[-2]]
    if mask is not None:
        allowed = allowed & np.asarray(mask, dtype=bool)
    scores = np.where(allowed, scores, -np.inf)
    return softmax(scores, axis=-1) @ v


def split_heads(x, n_heads):
    n, d_model = x.shape
    if d_model % n_heads:
        raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
    return x.reshape(n, n_heads, d_model // n_heads).transpose(1, 0, 2)


def merge_heads(x):
    h, n, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(n, h * d_head)


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=False):
    q, k, v = (split_heads(x @ W, n_heads) for W in (Wq, Wk, Wv))
    heads = scaled_dot_product_attention(q, k, v, is_causal=causal)
    return merge_heads(heads) @ Wo


def future_leak(x, Wq, Wk, Wv, Wo, n_heads, causal):
    changed = x.copy()
    changed[-1] = changed[-1] + 10.0
    a = multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal)
    b = multi_head_attention(changed, Wq, Wk, Wv, Wo, n_heads, causal)
    return float(np.abs(a[:-1] - b[:-1]).max())


if __name__ == "__main__":
    import torch

    rng = np.random.default_rng(0)
    n, d_model, n_heads = 6, 16, 4
    x = rng.normal(size=(n, d_model))
    Wq, Wk, Wv, Wo = (rng.normal(size=(d_model, d_model)) / 4 for _ in range(4))
    out = multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=True)
    mha = torch.nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.from_numpy(np.concatenate([Wq.T, Wk.T, Wv.T])))
        mha.out_proj.weight.copy_(torch.from_numpy(Wo.T))
        t = torch.from_numpy(x)[None]
        ref, _ = mha(t, t, t, attn_mask=torch.nn.Transformer.generate_square_subsequent_mask(n, dtype=torch.float64), need_weights=False)
    print("output shape:", out.shape)
    print("max abs diff vs torch MultiheadAttention (causal):", f"{np.abs(out - ref[0].numpy()).max():.1e}")
    print("future leak without mask:", round(future_leak(x, Wq, Wk, Wv, Wo, n_heads, causal=False), 3))
    print("future leak with causal mask:", future_leak(x, Wq, Wk, Wv, Wo, n_heads, causal=True))
    print("softmax of a fully masked row:", softmax(np.array([-np.inf, -np.inf, -np.inf])))
