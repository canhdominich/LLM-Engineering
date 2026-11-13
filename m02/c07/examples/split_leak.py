import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab" / "solution"))
from intent_lora import DATA, accuracy, macro_f1, tfidf_baseline

df = pd.read_csv(DATA)
official_train, official_test = df[df.split == "train"], df[df.split == "test"]
rand_train, rand_test = train_test_split(df, test_size=63, stratify=df.intent, random_state=0)

for name, tr, te in [("random row split", rand_train, rand_test), ("template split", official_train, official_test)]:
    shared = te.template_id.isin(set(tr.template_id)).mean()
    pred = tfidf_baseline(tr.text, tr.intent, te.text)
    print(f"{name:<17} test rows whose template is also in train: {shared:6.1%}   "
          f"acc {accuracy(list(te.intent), pred):.3f}  macro-F1 {macro_f1(list(te.intent), pred):.3f}")

example = rand_test[rand_test.template_id.isin(set(rand_train.template_id))].iloc[0]
twin = rand_train[rand_train.template_id == example.template_id].iloc[0]
print(f"test : {example.text}")
print(f"train: {twin.text}")
