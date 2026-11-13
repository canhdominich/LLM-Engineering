from datetime import date

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

from _policy import POLICY

MODEL = "qwen3:1.7b"
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=600)
question = "Phí giao nội quận của Giao Hàng Anh Tí hiện nay là bao nhiêu?"


def retrieve(query, docs, k=1):
    vec = TfidfVectorizer().fit(docs + [query])
    scores = (vec.transform(docs) @ vec.transform([query]).T).toarray().ravel()
    return [docs[i] for i in scores.argsort()[::-1][:k]]


def ask(prompt):
    r = client.responses.create(model=MODEL, input=prompt, temperature=0, max_output_tokens=120,
                                reasoning={"effort": "none"})
    return r.output_text.strip(), r.usage.input_tokens


context = retrieve(question, POLICY)
plain, n_plain = ask(question + " Trả lời một câu ngắn.")
rag_prompt = f"Chỉ dùng thông tin sau để trả lời.\nThông tin: {context[0]}\nCâu hỏi: {question} Trả lời một câu ngắn."
rag, n_rag = ask(rag_prompt)
print(f"model: {MODEL}, run on {date.today()}")
print(f"retrieved: {context[0]}")
print(f"[prompt only, {n_plain} input tokens] {plain}")
print(f"[prompt + retrieved doc, {n_rag} input tokens] {rag}")
