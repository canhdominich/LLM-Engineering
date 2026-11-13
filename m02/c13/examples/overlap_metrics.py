import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from sentence_transformers import SentenceTransformer

from checks import fact_check, policy_check
from golden import by_id, fold, load_golden


def token_f1(a, b):
    ta, tb = Counter(fold(a).split()), Counter(fold(b).split())
    common = sum((ta & tb).values())
    if not common:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


CASES = {
    "G06": {"paraphrase": "Khoảng 25 phút nữa shipper sẽ giao tới bạn, mã đơn AT046354 nhé.",
            "wrong_minutes": "Đơn AT046354 dự kiến tới trong khoảng 15 phút nữa.",
            "no_answer": "Đơn AT046354 sẽ tới sớm, bạn chờ thêm chút nhé."},
    "G22": {"paraphrase": "Đơn AT043149 đang trên đường giao nên không đổi được địa chỉ nữa, mình nối bạn với nhân viên nhé.",
            "wrong_claim": "Mình đã đổi địa chỉ đơn AT043149 sang Đồi Thông cho bạn rồi nhé.",
            "no_answer": "Bạn chờ chút, mình kiểm tra đơn AT043149 nhé."},
}
golden = by_id(load_golden())
model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
for gid, candidates in CASES.items():
    item, ref = golden[gid], golden[gid]["reference"]
    vecs = model.encode(["passage: " + ref] + ["passage: " + c for c in candidates.values()], normalize_embeddings=True)
    print(f"{gid} question: {item['turns'][-1]['content']}")
    print("    reference:", ref)
    print(f"    {'candidate':14}{'token_f1':>9}{'cosine':>8}  rule_check")
    for (name, text), v in zip(candidates.items(), vecs[1:]):
        problems = fact_check(item, text) + policy_check(item, text)
        verdict = "pass" if not problems else "fail " + str(problems)
        print(f"    {name:14}{token_f1(text, ref):9.3f}{float(v @ vecs[0]):8.3f}  {verdict}")
