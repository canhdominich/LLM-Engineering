import unicodedata
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
msgs = pd.read_csv(DATA)
msgs["clean"] = msgs["text"].map(lambda t: unicodedata.normalize("NFC", t).lower())
train = msgs.loc[msgs["split"] == "train", "clean"].tolist()
test = msgs.loc[msgs["split"] == "test", "clean"].tolist()

for ch in ["a", "ư", "đ", "ệ"]:
    print(f"{ch!r}: {len(ch.encode('utf-8'))} bytes {ch.encode('utf-8').hex(' ')}")
text = "".join(test)
print(f"test set: {len(text)} characters, {len(text.encode('utf-8'))} bytes")


def build(byte_level, size=400):
    tok = Tokenizer(models.BPE(unk_token=None if byte_level else "[UNK]"))
    if byte_level:
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        alphabet = pre_tokenizers.ByteLevel.alphabet()
    else:
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        alphabet = []
    trainer = trainers.BpeTrainer(vocab_size=size, initial_alphabet=alphabet,
                                  special_tokens=[] if byte_level else ["[UNK]"], show_progress=False)
    tok.train_from_iterator(train, trainer)
    return tok


for name, tok in [("char-level", build(False)), ("byte-level", build(True))]:
    avg = sum(len(tok.encode(t).ids) for t in test) / len(test)
    new = tok.encode("cảm ơn 🙏")
    print(f"{name}: vocab={tok.get_vocab_size()} tokens/msg={avg:.1f}  'cảm ơn 🙏' has {len(new.ids)} tokens, unknown={new.tokens.count('[UNK]')}, decodes to {tok.decode(new.ids)!r}")
