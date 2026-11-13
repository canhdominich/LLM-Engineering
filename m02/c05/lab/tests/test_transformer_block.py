import pytest
import torch
from torch import nn
import torch.nn.functional as F

import transformer_block as tb


def make(d_model=32, n_heads=4, d_ff=64, seed=0):
    torch.manual_seed(seed)
    return tb.TransformerBlock(d_model, n_heads, d_ff).eval()


def test_sinusoidal_shape_and_position_zero():
    P = tb.sinusoidal_positions(16, 8)
    assert P.shape == (16, 8)
    assert torch.allclose(P[0, 0::2], torch.zeros(4))
    assert torch.allclose(P[0, 1::2], torch.ones(4))
    assert P[1, 0].item() == pytest.approx(0.8414709, abs=1e-5)


def test_sinusoidal_dot_depends_only_on_offset():
    P = tb.sinusoidal_positions(128, 16)
    assert (P[2] @ P[5]).item() == pytest.approx((P[100] @ P[103]).item(), abs=1e-3)
    with pytest.raises(ValueError):
        tb.sinusoidal_positions(4, 7)


def test_causal_mask_is_lower_triangular_bool():
    m = tb.causal_mask(4)
    assert m.dtype == torch.bool
    assert m.tolist() == [[True, False, False, False], [True, True, False, False], [True, True, True, False], [True, True, True, True]]


def test_attention_shape_and_head_check():
    attn = tb.MultiHeadSelfAttention(32, 4)
    assert attn(torch.randn(3, 7, 32)).shape == (3, 7, 32)
    with pytest.raises(ValueError):
        tb.MultiHeadSelfAttention(10, 4)


@pytest.mark.parametrize("causal", [False, True])
def test_attention_matches_torch_reference(causal):
    torch.manual_seed(1)
    attn = tb.MultiHeadSelfAttention(32, 4)
    x = torch.randn(2, 6, 32)
    q, k, v = F.linear(x, attn.qkv.weight, attn.qkv.bias).chunk(3, dim=-1)
    split = lambda t: t.view(2, 6, 4, 8).transpose(1, 2)
    y = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=causal)
    ref = attn.proj(y.transpose(1, 2).reshape(2, 6, 32))
    assert torch.allclose(attn(x, causal=causal), ref, atol=1e-5)


def test_causal_attention_ignores_future_tokens():
    torch.manual_seed(2)
    attn = tb.MultiHeadSelfAttention(32, 4)
    x = torch.randn(1, 6, 32)
    x2 = x.clone()
    x2[0, 5] = torch.randn(32)
    assert torch.allclose(attn(x, causal=True)[0, :5], attn(x2, causal=True)[0, :5], atol=1e-6)
    assert not torch.allclose(attn(x)[0, :5], attn(x2)[0, :5], atol=1e-4)


def test_feedforward_works_on_each_token_alone():
    torch.manual_seed(3)
    ff = tb.FeedForward(32, 64)
    x = torch.randn(1, 5, 32)
    x2 = x.clone()
    x2[0, 2] = torch.randn(32)
    diff = (ff(x) - ff(x2)).abs().amax(dim=-1)[0]
    assert diff[2] > 1e-3
    assert torch.allclose(diff[[0, 1, 3, 4]], torch.zeros(4))


def test_block_keeps_shape_and_residual_passes_input_through():
    block = make()
    x = torch.randn(2, 5, 32)
    assert block(x).shape == x.shape
    with torch.no_grad():
        for layer in (block.attn.proj, block.ff.down):
            layer.weight.zero_()
            layer.bias.zero_()
    assert torch.allclose(block(x), x)


@pytest.mark.parametrize("causal", [False, True])
def test_block_matches_torch_pre_norm_layer(causal):
    block = make(seed=4)
    ref = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, activation="gelu", batch_first=True, norm_first=True).eval()
    with torch.no_grad():
        ref.self_attn.in_proj_weight.copy_(block.attn.qkv.weight)
        ref.self_attn.in_proj_bias.copy_(block.attn.qkv.bias)
        ref.self_attn.out_proj.weight.copy_(block.attn.proj.weight)
        ref.self_attn.out_proj.bias.copy_(block.attn.proj.bias)
        ref.linear1.load_state_dict(block.ff.up.state_dict())
        ref.linear2.load_state_dict(block.ff.down.state_dict())
        ref.norm1.load_state_dict(block.ln1.state_dict())
        ref.norm2.load_state_dict(block.ln2.state_dict())
    x = torch.randn(2, 6, 32)
    mask = nn.Transformer.generate_square_subsequent_mask(6) if causal else None
    with torch.no_grad():
        expected = ref(x, src_mask=mask, is_causal=causal)
        got = block(x, causal=causal)
    assert torch.allclose(got, expected, atol=1e-5)


@pytest.mark.parametrize("d_model,n_heads,d_ff", [(32, 4, 64), (64, 4, 256), (128, 8, 512)])
def test_param_formula_matches_module_and_torch(d_model, n_heads, d_ff):
    block = tb.TransformerBlock(d_model, n_heads, d_ff)
    ref = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
    n = tb.count_params(block)
    assert n == tb.block_param_formula(d_model, d_ff)
    assert n == sum(p.numel() for p in ref.parameters())
