import unicodedata

import pytest

import intent_baseline as ib

TINY_TEXTS = ["hoàn tiền giúp tôi", "hoan tien di", "khi nào tới", "bao giờ giao tới", "đổi địa chỉ nhà", "doi dia chi"]
TINY_LABELS = ["refund", "refund", "eta", "eta", "change_address", "change_address"]


@pytest.fixture(scope="module")
def data():
    return ib.load_split()


@pytest.fixture(scope="module")
def fitted(data):
    train, _ = data
    return ib.build_pipeline().fit(train["text"], train["intent"])


def test_normalize_removes_diacritics_and_d():
    assert ib.normalize_text("Đơn hàng ĐÃ giao") == "don hang da giao"


def test_normalize_same_for_nfc_and_nfd():
    text = "Hoàn tiền đơn trễ"
    assert ib.normalize_text(unicodedata.normalize("NFD", text)) == ib.normalize_text(unicodedata.normalize("NFC", text))


def test_normalize_masks_order_id_any_case():
    a = ib.normalize_text("Đơn AT012345 ở đâu")
    b = ib.normalize_text("don at999999 o dau")
    assert a == b == "don orderid o dau"


def test_normalize_collapses_spaces():
    assert ib.normalize_text("  giao   gấp  ") == "giao gap"


def test_pipeline_steps_and_normalized_vocabulary():
    pipe = ib.build_pipeline(C=3.0).fit(TINY_TEXTS, TINY_LABELS)
    assert list(pipe.named_steps) == ["tfidf", "clf"]
    assert pipe.named_steps["clf"].C == 3.0
    vocab = pipe.named_steps["tfidf"].vocabulary_
    assert "hoan" in vocab and "hoàn" not in vocab


def test_pipeline_raw_keeps_diacritics():
    pipe = ib.build_pipeline(normalize=False).fit(TINY_TEXTS, TINY_LABELS)
    assert "hoàn" in pipe.named_steps["tfidf"].vocabulary_


def test_evaluate_keys_and_confusion_shape():
    pipe = ib.build_pipeline().fit(TINY_TEXTS, TINY_LABELS)
    r = ib.evaluate(pipe, TINY_TEXTS, TINY_LABELS)
    assert set(r) >= {"accuracy", "macro_f1", "confusion"}
    assert r["confusion"].shape == (6, 6)
    assert list(r["confusion"].index) == ib.LABELS
    assert int(r["confusion"].to_numpy().sum()) == len(TINY_TEXTS)
    assert r["accuracy"] == pytest.approx(1.0)


def test_evaluate_macro_f1_counts_all_labels():
    class Always:
        def predict(self, texts):
            return ["refund"] * len(texts)

    r = ib.evaluate(Always(), ["a", "b"], ["refund", "eta"])
    assert r["accuracy"] == pytest.approx(0.5)
    assert r["macro_f1"] == pytest.approx((2 / 3) / 6)


def test_top_terms_refund_are_money_words(fitted):
    terms = ib.top_terms(fitted, "refund", k=5)
    assert len(terms) == 5
    assert "tien" in terms or "hoan" in terms


def test_fallback_threshold_extremes(fitted, data):
    _, test = data
    texts = test["text"].head(10)
    assert ib.predict_with_fallback(fitted, texts, threshold=1.01) == [ib.HUMAN] * 10
    assert ib.predict_with_fallback(fitted, texts, threshold=0.0) == list(fitted.predict(list(texts)))


def test_normalized_beats_raw_on_template_split(data):
    train, test = data
    raw = ib.build_pipeline(normalize=False).fit(train["text"], train["intent"])
    norm = ib.build_pipeline().fit(train["text"], train["intent"])
    f_raw = ib.evaluate(raw, test["text"], test["intent"])["macro_f1"]
    f_norm = ib.evaluate(norm, test["text"], test["intent"])["macro_f1"]
    assert f_norm >= 0.70
    assert f_norm > f_raw
