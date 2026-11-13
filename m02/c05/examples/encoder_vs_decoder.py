import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
df = pd.read_csv(DATA)
train, test = df[df.split == "train"], df[df.split == "test"]
LABELS = sorted(df.intent.unique())

encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
t0 = time.perf_counter()
E_train = encoder.encode(["query: " + t for t in train.text], normalize_embeddings=True)
E_test = encoder.encode(["query: " + t for t in test.text], normalize_embeddings=True)
ms = (time.perf_counter() - t0) * 1000 / len(df)
nearest = (E_test @ E_train.T).argmax(axis=1)  # láng giềng gần nhất theo cosine
acc = (train.intent.to_numpy()[nearest] == test.intent.to_numpy()).mean()
print(f"encoder e5-small + 1-NN: accuracy {acc:.3f} on {len(test)} test messages, about {ms:.0f} ms per message")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=300)
rule = "Classify the customer message into one intent: " + ", ".join(LABELS) + ". Reply with the intent only."
sample = test.groupby("intent").head(1)  # mỗi intent một tin nhắn
hits, secs = 0, []
for text, gold in zip(sample.text, sample.intent):
    t0 = time.perf_counter()
    r = client.responses.create(model="qwen3:1.7b", instructions=rule, input=text,
                                reasoning={"effort": "none"}, max_output_tokens=20, temperature=0)
    secs.append(time.perf_counter() - t0)
    answer = r.output_text.strip()
    hits += answer == gold
    print(f"  gold={gold:<15} decoder={answer!r}")
print(f"decoder qwen3:1.7b: {hits}/{len(sample)} exact labels, about {sum(secs) / len(secs):.1f} s per message")
