from _data import sample24
from _fewshot import fixed_six, nearest

print("fixed examples (first train message of each intent):")
for text, label in fixed_six():
    print(f"  {label:15} {text[:52]}")
rows = sample24().set_index("message_id")
for mid in ["M0130", "M0091", "M0164"]:
    print(f"\n{mid} [{rows.at[mid, 'intent']}] {rows.at[mid, 'text']}")
    for text, label, score in nearest(rows.at[mid, "text"]):
        print(f"  {score:.3f} {label:15} {text[:52]}")
