import random

from bigram_counts import count_bigrams, load_sentences, next_word_probs

following = count_bigrams(load_sentences())


def generate(start: str, greedy: bool, temperature: float = 1.0, max_words: int = 12, seed: int = 0) -> str:
    rng = random.Random(seed)
    words = [start]
    while words[-1] != "</s>" and len(words) < max_words:
        probs = next_word_probs(following, words[-1])  # chỉ nhìn đúng một từ cuối
        if greedy:
            nxt = max(probs, key=probs.get)
        else:
            weights = [p ** (1 / temperature) for p in probs.values()]
            nxt = rng.choices(list(probs), weights=weights)[0]
        words.append(nxt)
    return " ".join(words)


for start in ["đơn", "khi", "shipper"]:
    print("greedy:", generate(start, greedy=True))
for seed in range(3):
    print(f"sample {seed}, t=1.0:", generate("đơn", greedy=False, seed=seed))
for seed in range(2):
    print(f"sample {seed}, t=0.3:", generate("đơn", greedy=False, temperature=0.3, seed=seed))

words, trace = ["khi"], []
while words[-1] != "</s>":
    probs = next_word_probs(following, words[-1])
    best = max(probs, key=probs.get)
    trace.append(f"{best} ({probs[best]:.2f})")
    words.append(best)
print("greedy trace from 'khi':", ", ".join(trace))
