import time

from _providers import Delta, Done, make_provider

SYSTEM = {"role": "system", "content": "Bạn là trợ lý chat của Giao Hang Anh Ti. Trả lời một câu ngắn bằng tiếng Việt."}


def answer(provider, question):  # code nghiệp vụ: chỉ biết Delta và Done
    t0, text, first = time.perf_counter(), "", None
    for item in provider.stream([SYSTEM, {"role": "user", "content": question}], max_tokens=80):
        if isinstance(item, Delta):
            first = first or time.perf_counter() - t0
            text += item.text
        elif isinstance(item, Done):
            usage = (item.input_tokens, item.output_tokens, item.model)
    return text.strip(), usage, first, time.perf_counter() - t0


for env in ({"LLM_PROVIDER": "fake", "FAKE_REPLY": "Đơn AT047838 đang được giao ở Cho Moi."}, {"LLM_PROVIDER": "ollama"}):
    p = make_provider(env)
    text, (tin, tout, model), first, total = answer(p, "Đơn AT047838 đang ở đâu?")
    print(f"[{p.name}] model={model} in={tin} out={tout} first={first:.2f} s total={total:.2f} s")
    print(f"   {text}")
