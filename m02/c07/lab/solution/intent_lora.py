import json
import math
import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

DATA = Path(os.environ.get("ANHTI_M02_DATA", Path(__file__).resolve().parents[3] / "data")) / "messages.csv"
LABELS = ["change_address", "complaint_late", "eta", "order_status", "other", "refund"]
MODEL_NAME = "intfloat/multilingual-e5-small"


def load_messages(path=DATA):
    df = pd.read_csv(path)
    df["text"] = df["text"].map(lambda t: unicodedata.normalize("NFC", t))
    return df[df.split == "train"], df[df.split == "test"]


def normalize_text(text):
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn").replace("đ", "d")
    return " ".join(re.sub(r"at\d{6}", " orderid ", text).split())


def tfidf_baseline(train_texts, train_labels, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(TfidfVectorizer(preprocessor=normalize_text), LogisticRegression(C=10.0, max_iter=1000))
    clf.fit(list(train_texts), list(train_labels))
    return list(clf.predict(list(test_texts)))


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(model, targets=("query", "value"), r=8, alpha=16, train_names=("classifier",)):
    for p in model.parameters():
        p.requires_grad_(False)
    found = [(name, mod) for name, mod in model.named_modules()
             if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets]
    for name, mod in found:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.split(".")[-1], LoRALinear(mod, r, alpha))
    for name, p in model.named_parameters():
        if any(t in name for t in train_names):
            p.requires_grad_(True)
    return len(found)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def parse_label(reply, labels=LABELS, fallback="other"):
    text = re.sub(r"<think>.*?</think>", " ", reply, flags=re.S).lower()
    hits = [(text.find(label), label) for label in labels if label in text]
    return min(hits)[1] if hits else fallback


def build_prompt(message, labels=LABELS):
    return ("Phân loại ý định của tin nhắn khách hàng gửi Giao Hàng Anh Tí. "
            f"Chọn đúng một nhãn trong: {', '.join(labels)}. Chỉ trả về nhãn.\n"
            f"Tin nhắn: {message}")


def prompt_classify(messages, ask, labels=LABELS):
    return [parse_label(ask(build_prompt(m, labels)), labels) for m in messages]


def macro_f1(y_true, y_pred, labels=LABELS):
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


class E5Classifier(nn.Module):
    def __init__(self, encoder, n_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        # trung bình các token thật (mean pooling) như cách e5 được huấn luyện
        return self.classifier((h * mask).sum(1) / mask.sum(1))


def train_lora(train_texts, train_labels, epochs=4, r=8, alpha=16, lr=3e-3, seed=0):
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = E5Classifier(AutoModel.from_pretrained(MODEL_NAME), len(LABELS))
    add_lora(model, r=r, alpha=alpha)
    encode = lambda texts: tok(["query: " + t for t in texts], padding=True, truncation=True, max_length=48, return_tensors="pt")
    y = torch.tensor([LABELS.index(label) for label in train_labels])
    texts = list(train_texts)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(texts)).split(16):
            batch = encode([texts[i] for i in idx])
            loss = nn.functional.cross_entropy(model(batch["input_ids"], batch["attention_mask"]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()

    @torch.no_grad()
    def predict(batch_texts):
        out = []
        for i in range(0, len(batch_texts), 32):
            batch = encode(list(batch_texts[i:i + 32]))
            out += model(batch["input_ids"], batch["attention_mask"]).argmax(-1).tolist()
        return [LABELS[i] for i in out]

    return model, predict


def ollama_ask(model="qwen3:1.7b", cache_file=Path(__file__).with_name("prompt_cache.json")):
    from openai import OpenAI

    client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"), api_key="ollama", timeout=600)
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

    def ask(prompt):
        # lưu câu trả lời theo prompt để chạy lại không phải gọi model lần nữa
        if prompt not in cache:
            r = client.responses.create(model=model, input=prompt, temperature=0, max_output_tokens=20,
                                        reasoning={"effort": "none"})
            cache[prompt] = r.output_text
            cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        return cache[prompt]

    return ask


def main(use_llm=True):
    train, test = load_messages()
    y_true = list(test.intent)
    rows = [("TF-IDF + logistic", "-", tfidf_baseline(train.text, train.intent, test.text))]
    model, predict = train_lora(train.text, train.intent)
    trainable, total = count_params(model)
    rows.append(("e5-small + LoRA r=8", f"{trainable:,} / {total:,}", predict(list(test.text))))
    raw = []
    if use_llm:
        ask = ollama_ask()
        pred = prompt_classify(list(test.text), lambda p: raw.append(ask(p)) or raw[-1])
        rows.append(("qwen3:1.7b prompt", "0", pred))
    print(f"train {len(train)} messages, test {len(test)} messages (held-out templates)")
    for name, params, pred in rows:
        print(f"{name:<20} acc {accuracy(y_true, pred):.3f}  macro-F1 {macro_f1(y_true, pred):.3f}  trainable {params}")
    print("per-label F1:", "  ".join(label[:12] for label in LABELS))
    for name, _, pred in rows:
        print(f"  {name[:18]:<18}", "  ".join(f"{macro_f1(y_true, pred, [label]):.2f}".rjust(len(label[:12])) for label in LABELS))
    if raw:
        invalid = sum(not any(label in r.lower() for label in LABELS) for r in raw)
        print(f"prompt replies with no valid label (counted as other): {invalid} / {len(raw)}")
    errors = [(t, p, m) for t, p, m in zip(y_true, rows[1][2], test.text) if t != p]
    print(f"LoRA errors: {len(errors)}")
    for t, p, m in errors[:6]:
        print(f"  true={t:<15} pred={p:<15} {m}")


if __name__ == "__main__":
    main(use_llm="--no-llm" not in sys.argv)
