# Data source: Truyen Kieu (Nguyen Du)

- File: `kieu.txt`, 3,254 verse lines, 104,204 characters, 132 distinct characters (Unicode NFC).
- Source page: https://vi.wikisource.org/wiki/Truy%E1%BB%87n_Ki%E1%BB%81u
- Exact revision used: oldid 205132 (2026-05-06T04:49:24Z), raw wikitext
  https://vi.wikisource.org/w/index.php?title=Truy%E1%BB%87n_Ki%E1%BB%81u&oldid=205132&action=raw
- Licence: the poem by Nguyen Du (1766-1820) is in the public domain; the Wikisource page carries the `{{PD-old}}` template. Wikisource site text is available under CC BY-SA 4.0; we keep only the verse lines and record the source here.
- Cleaning (`prepare_kieu.py`): keep the text between `<poem>` and `</poem>`, drop the `{{số|N}}` line-number templates and tabs, normalise to NFC, collapse spaces, one verse per line.
- Reproduce: `uv run python prepare_kieu.py` (needs internet; Wikimedia requires a User-Agent header).
