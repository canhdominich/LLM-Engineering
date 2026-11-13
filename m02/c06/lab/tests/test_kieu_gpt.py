import math

import pytest
import torch
import torch.nn.functional as F

from kieu_gpt import CausalSelfAttention, CharTokenizer, Config, GPT, generate, get_batch, sample_next, train


def small_model(vocab=12, block=8, dropout=0.0):
    torch.manual_seed(0)
    return GPT(Config(vocab_size=vocab, block_size=block, n_layer=2, n_head=2, n_embd=16, dropout=dropout)).eval()


def test_tokenizer_roundtrip():
    tok = CharTokenizer("Trăm năm trong cõi người ta,")
    assert tok.decode(tok.encode("năm người")) == "năm người"


def test_get_batch_shifts_by_one():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4, generator=torch.Generator().manual_seed(0))
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y, x + 1)
    assert int(y.max()) < 100


def test_attention_matches_sdpa():
    cfg = Config(vocab_size=10, block_size=8, n_head=2, n_embd=16, dropout=0.0)
    torch.manual_seed(0)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 6, 16)
    q, k, v = attn.qkv(x).split(16, dim=2)
    q, k, v = (t.view(2, 6, 2, 8).transpose(1, 2) for t in (q, k, v))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 6, 16)
    assert torch.allclose(attn(x), attn.proj(ref), atol=1e-5)


def test_future_token_does_not_change_past_logits():
    model = small_model()
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[1, 2, 3, 4, 5, 9]])
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[:, :5], lb[:, :5], atol=1e-6)
    assert not torch.allclose(la[:, 5], lb[:, 5])


def test_logits_shape_and_initial_loss():
    model = small_model(vocab=40)
    idx = torch.randint(40, (3, 8), generator=torch.Generator().manual_seed(1))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 8, 40)
    assert abs(loss.item() - math.log(40)) < 0.3


def test_forward_rejects_sequence_longer_than_block():
    with pytest.raises(ValueError):
        small_model(block=8)(torch.zeros(1, 9, dtype=torch.long))


def test_temperature_zero_and_top_k_one_are_greedy():
    logits = torch.tensor([[0.1, 2.0, 0.5, 1.9]])
    assert sample_next(logits, temperature=0).item() == 1
    gen = torch.Generator().manual_seed(0)
    assert all(sample_next(logits, 1.0, top_k=1, generator=gen).item() == 1 for _ in range(20))


def test_top_p_keeps_smallest_set_reaching_p():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    gen = torch.Generator().manual_seed(0)
    picks = {sample_next(logits, 1.0, top_p=0.7, generator=gen).item() for _ in range(300)}
    assert picks == {0, 1}
    picks = {sample_next(logits, 1.0, top_p=0.5, generator=gen).item() for _ in range(100)}
    assert picks == {0}


def test_temperature_changes_spread():
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0]])
    gen = torch.Generator().manual_seed(0)
    cold = sum(sample_next(logits, 0.2, generator=gen).item() == 0 for _ in range(500)) / 500
    hot = sum(sample_next(logits, 5.0, generator=gen).item() == 0 for _ in range(500)) / 500
    assert cold > 0.95
    assert hot < 0.45


def test_generate_runs_past_block_size():
    model = small_model(block=8)
    out = generate(model, torch.tensor([[1, 2, 3]]), 20, temperature=0)
    assert out.shape == (1, 23)
    assert torch.equal(out[:, :3], torch.tensor([[1, 2, 3]]))


def test_training_lowers_validation_loss():
    torch.manual_seed(0)
    data = torch.tensor([0, 1, 2, 3, 4, 5] * 60)
    model = GPT(Config(vocab_size=6, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0))
    history = train(model, data[:300], data[300:], steps=120, lr=1e-2, batch_size=16, eval_every=60)
    assert history[0][2] > 1.5
    assert history[-1][2] < 0.5
