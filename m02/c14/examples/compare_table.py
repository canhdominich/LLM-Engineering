from _bench import records, summary

ROWS = [("rules", "keyword rules"), ("tfidf", "TF-IDF + logistic"), ("gemma3-full", "gemma3:270m, full prompt"),
        ("qwen3-short", "qwen3:1.7b, short prompt"), ("qwen3-full", "qwen3:1.7b, full prompt")]

print(f"{'candidate':27}{'acc':>6}{'macroF1':>8}{'invalid':>8}{'p50 s':>8}{'p95 s':>8}{'in tok':>7}{'out tok':>8}")
for key, label in ROWS:
    s = summary(records(key))
    print(f"{label:27}{s['acc']:>6.3f}{s['f1']:>8.3f}{s['invalid']:>8}{s['p50']:>8.3f}{s['p95']:>8.3f}"
          f"{s['in_tok']:>7.0f}{s['out_tok']:>8.1f}")
print("golden set: 63 test messages (template split), thinking off for every LLM row")
