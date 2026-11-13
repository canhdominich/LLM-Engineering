from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"

RULES = [  # luật viết tay, xét theo thứ tự: luật đầu tiên khớp sẽ thắng
    ("refund", ["hoàn tiền", "trả lại tiền", "hoàn phí", "lại tiền"]),
    ("change_address", ["đổi địa chỉ", "sửa địa chỉ", "địa chỉ"]),
    ("complaint_late", ["trễ", "chậm", "quá giờ"]),
    ("eta", ["khi nào", "mấy giờ", "bao lâu"]),
    ("order_status", ["ở đâu", "trạng thái", "tình trạng", "chưa"]),
]


def classify(text: str) -> str:
    lower = text.lower()
    for intent, keywords in RULES:
        if any(k in lower for k in keywords):
            return intent
    return "other"


if __name__ == "__main__":
    df = pd.read_csv(DATA)
    test = df[df["split"] == "test"].copy()
    test["pred"] = test["text"].map(classify)
    test["hit"] = test["pred"] == test["intent"]
    print(f"test messages: {len(test)}")
    print(f"accuracy, all: {test['hit'].mean():.3f}")
    for has, group in test.groupby("has_diacritics"):
        label = "with diacritics" if has else "no diacritics"
        print(f"accuracy, {label:15}: {group['hit'].mean():.3f} ({len(group)} messages)")
    print("wrong examples:")
    wrong = test[~test["hit"]]
    for _, r in pd.concat([wrong[~wrong["has_diacritics"]].head(2), wrong[wrong["has_diacritics"]].head(2)]).iterrows():
        print(f"  {r['text'][:48]!r:52} true={r['intent']:<14} pred={r['pred']}")
