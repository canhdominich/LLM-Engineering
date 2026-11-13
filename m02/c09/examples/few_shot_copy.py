from _data import parse_label
from _fewshot import TRAIN, fixed_six, with_examples
from _llm import ask
from _prompts import RULES, wrap

six = fixed_six()
wrong = [(t, "complaint_late" if lab == "refund" else lab) for t, lab in six]
example = [t for t, lab in six if lab == "refund"][0]
template = TRAIN.loc[TRAIN["text"] == example, "template_id"].iloc[0]
# tin cùng template với ví dụ bị gán nhầm (không tính chính ví dụ đó): đây là phép thử, không phải bộ đánh giá
probe = TRAIN[(TRAIN["template_id"] == template) & (TRAIN["text"] != example)].head(4)
print(f"probe: {len(probe)} train messages of template {template}, true label refund")
VARIANTS = {
    "6 fixed": with_examples(six),
    "6, one wrong": with_examples(wrong),
    "6, one wrong, no rule": with_examples(wrong).replace(RULES.split("\n")[0] + "\n", ""),
}
for name, instructions in VARIANTS.items():
    labels = [parse_label(ask(instructions, wrap(t))["text"]) for t in probe["text"]]
    print(f"{name:22} {labels}")
