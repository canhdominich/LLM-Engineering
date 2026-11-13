import numpy as np
from sklearn.metrics import f1_score

from _common import LABELS, golden, rules_classify, run_candidate, run_local, tfidf_model


def records(name):
    if name == "rules":
        return run_local(rules_classify, golden())
    if name == "tfidf":
        pipe = tfidf_model()
        return run_local(lambda t: str(pipe.predict([t])[0]), golden())
    return run_candidate(name)


def summary(recs):
    y = [r["intent"] for r in recs]
    p = [r["pred"] for r in recs]
    secs = [r["seconds"] for r in recs]
    return {"n": len(recs), "acc": float(np.mean([a == b for a, b in zip(y, p)])),
            "f1": float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
            "invalid": p.count("invalid"), "p50": float(np.percentile(secs, 50)),
            "p95": float(np.percentile(secs, 95)), "in_tok": float(np.mean([r["in_tok"] for r in recs])),
            "out_tok": float(np.mean([r["out_tok"] for r in recs]))}
