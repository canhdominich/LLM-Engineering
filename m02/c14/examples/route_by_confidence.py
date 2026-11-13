import numpy as np

from _common import golden, run_candidate, tfidf_model

g = golden()
pipe = tfidf_model()
proba = pipe.predict_proba(g["text"])
conf = proba.max(axis=1)  # xác suất của nhãn mạnh nhất
cheap = pipe.classes_[proba.argmax(axis=1)]
llm = {r["message_id"]: r for r in run_candidate("qwen3-full")}
strong = np.array([llm[i]["pred"] for i in g["message_id"]])
llm_s = np.array([llm[i]["seconds"] for i in g["message_id"]])
truth = g["intent"].to_numpy()

print(f"{'route to LLM if conf <':24}{'sent':>6}{'acc':>7}{'LLM s / 1000 msgs':>19}")
for th in (0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01):
    send = conf < th
    pred = np.where(send, strong, cheap)
    print(f"{th:<24}{send.sum():>4}/63{(pred == truth).mean():>7.3f}{1000 * (llm_s * send).sum() / len(g):>19.0f}")
send = conf < 0.5
print(f"tfidf alone on the {send.sum()} low-confidence messages: {(cheap[send] == truth[send]).mean():.3f}, "
      f"qwen3 on the same messages: {(strong[send] == truth[send]).mean():.3f}")
