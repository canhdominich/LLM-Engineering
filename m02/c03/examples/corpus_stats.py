from collections import Counter

from _text import load_messages, tokenize

messages = load_messages()
docs = [tokenize(m["text"]) for m in messages]
freq = Counter(w for d in docs for w in d)
late = [d for m, d in zip(messages, docs) if m["intent"] == "late"]

print(f"messages: {len(messages)}, tokens: {sum(freq.values())}, vocabulary: {len(freq)}")
print("tokens of message 1:", docs[0])
print(f"late complaints: {len(late)}")
for word in ["trễ", "chậm", "muộn", "lề_mề"]:
    n = sum(word in d for d in late)
    print(f"  containing {word!r:9}: {n:4d} ({n / len(late):.1%})")
print("spellings of cancel:", {w: freq[w] for w in ["hủy", "huỷ"]})
print("top 8 words:", freq.most_common(8))
