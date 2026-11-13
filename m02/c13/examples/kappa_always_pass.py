from sklearn.metrics import cohen_kappa_score

from _common import human_labels


def kappa(a, b):
    n = len(a)
    p_obs = sum(x == y for x, y in zip(a, b)) / n
    p_yes = (sum(a) / n) * (sum(b) / n)
    p_no = (1 - sum(a) / n) * (1 - sum(b) / n)
    p_exp = p_yes + p_no  # hai bên trùng nhau do tình cờ, chỉ vì tỉ lệ đạt của mỗi bên
    return p_obs, p_exp, (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else 0.0


h = list(human_labels().values())
print(f"human labels: {len(h)} replies, {sum(h)} pass, {len(h) - sum(h)} fail")
for name, judge in [("always_fail", [False] * len(h)), ("always_pass", [True] * len(h)),
                    ("copy_human_but_flip_5", [not x if i < 5 else x for i, x in enumerate(h)])]:
    p_obs, p_exp, k = kappa(h, judge)
    print(f"{name:22} agreement {p_obs:.3f}  chance {p_exp:.3f}  kappa {k:.3f}  sklearn {cohen_kappa_score(h, judge):.3f}")
