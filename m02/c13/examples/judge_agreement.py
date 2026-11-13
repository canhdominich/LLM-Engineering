from statistics import median

from sklearn.metrics import cohen_kappa_score

from _common import human_labels, load_golden, replies
from checks import score
from judge import judge_checklist, judge_holistic

items = load_golden()
human = human_labels()
rows = []
for v in ["v1", "v2"]:
    for it, (gid, a) in zip(items, replies(v, items).items()):
        rows.append({"key": (gid, v), "item": it, "reply": a["reply"], "human": human[(gid, v)],
                     "rules": score(it, a)["rules_ok"], "check": judge_checklist(it, a["reply"]),
                     "holi": judge_holistic(it, a["reply"]) if v == "v1" else None})


def report(name, subset, get):
    h, j = [r["human"] for r in subset], [get(r) for r in subset]
    agree = sum(x == y for x, y in zip(h, j)) / len(h)
    fp = sum(y and not x for x, y in zip(h, j))
    fn = sum(x and not y for x, y in zip(h, j))
    print(f"{name:20} n={len(h)}  agreement {agree:.3f}  kappa {cohen_kappa_score(h, j):6.3f}  "
          f"judge_pass_human_fail {fp:2}  judge_fail_human_pass {fn:2}")


v1 = [r for r in rows if r["key"][1] == "v1"]
print("same 40 replies of prompt v1:")
report("holistic score>=4", v1, lambda r: r["holi"]["pass"])
report("checklist judge", v1, lambda r: r["check"]["pass"])
report("keyword rules", v1, lambda r: r["rules"])
print("all 80 replies (v1 + v2):")
report("checklist judge", rows, lambda r: r["check"]["pass"])
report("keyword rules", rows, lambda r: r["rules"])
c = [r["check"] for r in rows]
print(f"checklist cost: median {median(x['in_tok'] for x in c)} input + {median(x['out_tok'] for x in c)} output tokens, "
      f"median {median(x['seconds'] for x in c):.1f} s per reply on CPU")
