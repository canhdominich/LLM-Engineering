import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

URL = "https://vi.wikisource.org/w/index.php?title=Truy%E1%BB%87n_Ki%E1%BB%81u&oldid=205132&action=raw"
HERE = Path(__file__).resolve().parent


def clean(raw):
    body = raw.split("<poem>", 1)[1].split("</poem>", 1)[0]
    lines = []
    for line in body.splitlines():
        line = re.sub(r"\{\{số\|\d+\}\}", "", line)
        line = unicodedata.normalize("NFC", line.replace("\t", " ")).strip()
        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)
    return "\n".join(lines) + "\n"


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "anhti-ai-labs/0.1 (course lab; contact via repo)"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    text = clean(raw)
    (HERE / "kieu.txt").write_text(text, encoding="utf-8")
    print("lines", text.count("\n"), "chars", len(text), "unique chars", len(set(text)))


if __name__ == "__main__":
    sys.exit(main())
