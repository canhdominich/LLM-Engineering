import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

LABELS = ["order_status", "eta", "complaint_late", "refund", "change_address", "other"]
HUMAN = "needs_human"
DATA = Path(os.environ.get("ANHTI_DATA", Path(__file__).resolve().parents[3] / "data")) / "messages.csv"
ORDER_ID = re.compile(r"\bat\d{6}\b")


def load_split(path: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    return df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)


def normalize_text(text: str) -> str:
    # TODO 1: chữ thường, bỏ dấu tiếng Việt (kể cả đ thành d), thay mã đơn AT + 6 số bằng từ 'orderid', gộp khoảng trắng
    raise NotImplementedError


def build_pipeline(normalize: bool = True, C: float = 10.0) -> Pipeline:
    # TODO 2: Pipeline gồm bước 'tfidf' (TfidfVectorizer, dùng normalize_text làm preprocessor khi normalize=True)
    # và bước 'clf' (LogisticRegression với C đã cho, max_iter=1000)
    raise NotImplementedError


def evaluate(pipe: Pipeline, texts, labels) -> dict:
    # TODO 3: trả về dict có 'accuracy', 'macro_f1' (trung bình trên đủ 6 nhãn LABELS)
    # và 'confusion' (DataFrame 6 x 6, hàng là nhãn thật, cột là nhãn dự đoán, theo thứ tự LABELS)
    raise NotImplementedError


def top_terms(pipe: Pipeline, intent: str, k: int = 5) -> list[str]:
    # TODO 4: k từ có hệ số logistic regression lớn nhất cho nhãn intent, giảm dần
    raise NotImplementedError


def predict_with_fallback(pipe: Pipeline, texts, threshold: float = 0.4) -> list[str]:
    # TODO 5: nhãn có xác suất cao nhất; nếu xác suất đó nhỏ hơn threshold thì trả về HUMAN
    raise NotImplementedError


def main() -> None:
    train, test = load_split()
    print(f"train {len(train)}, test {len(test)} (split by template)")
    for name, pipe in [("raw text", build_pipeline(normalize=False)), ("normalized", build_pipeline())]:
        pipe.fit(train["text"], train["intent"])
        r = evaluate(pipe, test["text"], test["intent"])
        print(f"{name:11} accuracy {r['accuracy']:.3f}  macro F1 {r['macro_f1']:.3f}  vocabulary {len(pipe.named_steps['tfidf'].vocabulary_)}")
    print("confusion (rows true, columns predicted), normalized:")
    print(r["confusion"].to_string())
    for intent in LABELS:
        print(f"top terms {intent:15}: {', '.join(top_terms(pipe, intent))}")
    routed = predict_with_fallback(pipe, test["text"], threshold=0.4)
    kept = [(p, t) for p, t in zip(routed, test["intent"]) if p != HUMAN]
    print(f"threshold 0.4: {len(test) - len(kept)} of {len(test)} sent to a human, accuracy on the rest {np.mean([p == t for p, t in kept]):.3f}")

    shuffled = pd.concat([train, test]).sample(frac=1.0, random_state=0).reset_index(drop=True)
    cut = int(len(shuffled) * 0.8)
    leak = build_pipeline().fit(shuffled["text"][:cut], shuffled["intent"][:cut])
    print(f"random row split instead: accuracy {evaluate(leak, shuffled['text'][cut:], shuffled['intent'][cut:])['accuracy']:.3f}")


if __name__ == "__main__":
    main()
