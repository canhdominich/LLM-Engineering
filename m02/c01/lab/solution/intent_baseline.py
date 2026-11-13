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
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d")
    text = ORDER_ID.sub(" orderid ", text)
    return " ".join(text.split())


def build_pipeline(normalize: bool = True, C: float = 10.0) -> Pipeline:
    vectorizer = TfidfVectorizer(preprocessor=normalize_text) if normalize else TfidfVectorizer()
    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", LogisticRegression(C=C, max_iter=1000)),
    ])


def evaluate(pipe: Pipeline, texts, labels) -> dict:
    pred = pipe.predict(list(texts))
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, labels=LABELS, average="macro", zero_division=0)),
        "confusion": pd.DataFrame(confusion_matrix(labels, pred, labels=LABELS), index=LABELS, columns=LABELS),
    }


def top_terms(pipe: Pipeline, intent: str, k: int = 5) -> list[str]:
    clf = pipe.named_steps["clf"]
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    row = clf.coef_[list(clf.classes_).index(intent)]
    order = np.argsort(-row, kind="stable")[:k]
    return [str(names[i]) for i in order]


def predict_with_fallback(pipe: Pipeline, texts, threshold: float = 0.4) -> list[str]:
    proba = pipe.predict_proba(list(texts))
    best = proba.argmax(axis=1)
    classes = pipe.classes_
    return [str(classes[i]) if p[i] >= threshold else HUMAN for i, p in zip(best, proba)]


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
