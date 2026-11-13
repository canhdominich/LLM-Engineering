import re
import unicodedata
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
msgs = pd.read_csv(DATA)
msgs["clean"] = msgs["text"].map(lambda t: unicodedata.normalize("NFC", t).lower())
train = msgs.loc[msgs["split"] == "train", "clean"].tolist()
test = msgs.loc[msgs["split"] == "test", "clean"].tolist()
print(f"train messages: {len(train)}  test messages: {len(test)}")


def words(text):
    return re.findall(r"\w+|[^\w\s]", text)


def report(name, vocab_size, pieces):
    total = sum(len(p) for p in pieces)
    unk = sum(t == "[UNK]" for p in pieces for t in p)
    print(f"{name:10} vocab={vocab_size:4}  tokens/msg={total / len(pieces):5.1f}  unknown={unk / total:6.1%}")


word_vocab = {w for t in train for w in words(t)}
word_pieces = [[w if w in word_vocab else "[UNK]" for w in words(t)] for t in test]
report("word", len(word_vocab), word_pieces)

char_vocab = {c for t in train for c in t}
report("character", len(char_vocab), [[c if c in char_vocab else "[UNK]" for c in t] for t in test])

for size in (150, 400, 1000):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(train, trainers.BpeTrainer(vocab_size=size, special_tokens=["[UNK]"], show_progress=False))
    report("BPE", tok.get_vocab_size(), [tok.encode(t).tokens for t in test])

unknown = [w for t in test for w in words(t) if w not in word_vocab]
codes = [w for w in unknown if re.fullmatch(r"at\d{6}", w)]
print(f"unknown words: {len(unknown)}  order codes: {len(codes)}  examples: {[w for w in unknown if w not in codes][:8]}")

msg = test[3]
print("message:", msg)
print("  word:", word_pieces[3])
print("  BPE :", tok.encode(msg).tokens)
