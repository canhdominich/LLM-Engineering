import pytest
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import BertConfig, BertForSequenceClassification

from common.llm import FakeLLM
from intent_lora import (LABELS, LoRALinear, add_lora, build_prompt, count_params, macro_f1, parse_label,
                         prompt_classify, tfidf_baseline)


def tiny_bert():
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=32, num_labels=6)
    return BertForSequenceClassification(cfg).eval()


def test_tfidf_baseline_learns_simple_patterns():
    train = ["hủy đơn giúp tôi", "tôi muốn hủy đơn", "đơn tới đâu rồi", "đơn đang ở đâu", "hoàn tiền cho tôi", "trả lại tiền giúp"]
    labels = ["cancel", "cancel", "status", "status", "refund", "refund"]
    pred = tfidf_baseline(train, labels, ["hủy đơn này", "đơn đâu rồi", "xin hoàn tiền"])
    assert pred == ["cancel", "status", "refund"]


def test_lora_linear_starts_equal_to_base():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    layer = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)
    assert torch.allclose(layer(x), base(x))
    assert layer.scale == 2.0


def test_lora_linear_trains_only_a_and_b():
    base = nn.Linear(16, 24)
    layer = LoRALinear(base, r=4, alpha=8)
    trainable = [p for p in layer.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == 4 * (16 + 24)
    layer(torch.randn(3, 16)).sum().backward()
    assert base.weight.grad is None
    assert layer.B.grad.abs().sum() > 0
    assert layer.A.grad.abs().sum() == 0


def test_lora_linear_changes_output_after_update():
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    layer = LoRALinear(base, r=2, alpha=2)
    with torch.no_grad():
        layer.B.fill_(0.1)
    x = torch.randn(2, 8)
    expected = base(x) + (x @ layer.A.T @ layer.B.T) * 1.0
    assert torch.allclose(layer(x), expected, atol=1e-6)
    assert not torch.allclose(layer(x), base(x))


def test_add_lora_on_tiny_bert():
    model = tiny_bert()
    ids = torch.randint(0, 100, (2, 7))
    before = model(input_ids=ids).logits
    n = add_lora(model, targets=("query", "value"), r=4, alpha=8)
    assert n == 4
    assert isinstance(model.bert.encoder.layer[0].attention.self.query, LoRALinear)
    assert isinstance(model.bert.encoder.layer[1].attention.self.value, LoRALinear)
    assert not isinstance(model.bert.encoder.layer[0].attention.self.key, LoRALinear)
    assert torch.allclose(model(input_ids=ids).logits, before, atol=1e-6)
    trainable, total = count_params(model)
    lora = 4 * 4 * (32 + 32)
    classifier = 32 * 6 + 6
    assert trainable == lora + classifier
    assert total > trainable * 10


def test_add_lora_wrong_target_finds_nothing():
    model = tiny_bert()
    assert add_lora(model, targets=("q_proj", "v_proj")) == 0


def test_count_params_simple():
    m = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    m[0].weight.requires_grad_(False)
    assert count_params(m) == (3 + 6 + 2, 15 + 8)


@pytest.mark.parametrize("reply,expected", [
    ("refund", "refund"),
    ("<think>khách nói trễ nhưng đòi tiền</think>\nrefund", "refund"),
    ("Nhãn: ORDER_STATUS.", "order_status"),
    ("eta hoặc complaint_late", "eta"),
    ("mình không chắc", "other"),
])
def test_parse_label(reply, expected):
    assert parse_label(reply) == expected


def test_prompt_classify_with_fake_llm():
    fake = FakeLLM(["refund", "<think>...</think> eta"])
    ask = lambda prompt: fake.chat([{"role": "user", "content": prompt}]).text
    pred = prompt_classify(["hang vo roi, cho toi xin lai tien", "mấy giờ giao"], ask)
    assert pred == ["refund", "eta"]
    sent = fake.calls[0]["messages"][0]["content"]
    assert "hang vo roi, cho toi xin lai tien" in sent
    assert all(label in sent for label in LABELS)
    assert build_prompt("x").count("\n") >= 1


def test_macro_f1_matches_sklearn():
    y_true = ["eta", "eta", "refund", "other", "other", "order_status"]
    y_pred = ["eta", "refund", "refund", "other", "eta", "other"]
    ours = macro_f1(y_true, y_pred)
    ref = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    assert ours == pytest.approx(ref)
