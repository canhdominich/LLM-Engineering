import numpy as np

from _data import messages
from _prompts import BASE, wrap

TRAIN = messages().query("split == 'train'").sort_values("message_id").reset_index(drop=True)


def fixed_six():
    # ví dụ đầu tiên của mỗi ý định trong tập train
    return list(TRAIN.drop_duplicates("intent")[["text", "intent"]].itertuples(index=False, name=None))


def with_examples(examples, answer=lambda label: label):
    lines = [f"{wrap(text)}\n{answer(label)}" for text, label in examples]
    return BASE + "\nExamples:\n" + "\n".join(lines)


_model = None
_train_vecs = None


def nearest(text, k=3):
    global _model, _train_vecs
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
        _train_vecs = _model.encode(["query: " + t for t in TRAIN["text"]], normalize_embeddings=True)
    q = _model.encode(["query: " + text], normalize_embeddings=True)[0]
    scores = _train_vecs @ q
    top = np.argsort(-scores, kind="stable")[:k]
    return [(TRAIN.at[i, "text"], TRAIN.at[i, "intent"], float(scores[i])) for i in top]
