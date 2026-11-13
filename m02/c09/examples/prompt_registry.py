import hashlib
from pathlib import Path
from string import Template

from _data import LABELS
from _prompts import BASE

PROMPTS = Path(__file__).resolve().parents[1] / "data" / "prompts"


def load(name):
    text = (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
    version = hashlib.sha256(text.encode()).hexdigest()[:8]  # mã phiên bản theo nội dung
    return Template(text), version


if __name__ == "__main__":
    for name in ["classify_v4", "classify_v5"]:
        tpl, version = load(name)
        rendered = tpl.substitute(labels=", ".join(LABELS)).strip()
        print(f"{name}: version {version}, {len(rendered)} chars, equals v4 of lesson 1: {rendered == BASE}")

    tpl, _ = load("classify_v4")
    print("safe_substitute without labels ends with:", repr(tpl.safe_substitute()[-30:]))
    try:
        tpl.substitute()
    except KeyError as e:
        print("substitute without labels raises KeyError:", e)
