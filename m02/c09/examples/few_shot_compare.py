from _data import parse_label, sample12
from _fewshot import fixed_six, nearest, with_examples
from _llm import MODEL, ask
from _prompts import BASE, wrap

six = fixed_six()
wrong = [(t, "complaint_late" if lab == "refund" else lab) for t, lab in six]  # gán nhầm 1 nhãn
VARIANTS = {
    "zero-shot": lambda text: BASE,
    "6 fixed": lambda text: with_examples(six),
    "6, one wrong": lambda text: with_examples(wrong),
    "6 verbose": lambda text: with_examples(six, answer=lambda lab: f"The label is {lab}."),
    "3 nearest": lambda text: with_examples([(t, lab) for t, lab, _ in nearest(text)]),
}

rows = sample12()
preds = {}
print(f"{'variant':14}{'in tok':>7}{'out tok':>8}{'parsed':>7}{'exact':>6}{'refund ok':>10}")
for name, build in VARIANTS.items():
    parsed = exact = refund = n_in = n_out = 0
    preds[name] = []
    for _, r in rows.iterrows():
        out = ask(build(r["text"]), wrap(r["text"]))
        preds[name].append(out["text"])
        parsed += parse_label(out["text"]) == r["intent"]
        exact += out["text"] == r["intent"]
        refund += r["intent"] == "refund" and parse_label(out["text"]) == "refund"
        n_in += out["in_tok"]
        n_out += out["out_tok"]
    print(f"{name:14}{n_in / len(rows):>7.0f}{n_out / len(rows):>8.1f}{parsed:>7}{exact:>6}{refund:>8}/2")
print(f"{len(rows)} messages, model {MODEL}, reasoning off")
print("changed by the one wrong example:")
for i, r in rows.iterrows():
    a, b = parse_label(preds["6 fixed"][i]), parse_label(preds["6, one wrong"][i])
    if a != b:
        print(f"  {r['message_id']} [{r['intent']}] {a} to {b}: {r['text'][:40]}")
print("verbose reply sample:", repr(preds["6 verbose"][0][:60]))
