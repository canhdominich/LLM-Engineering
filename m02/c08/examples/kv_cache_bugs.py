import torch

from _tiny_gpt import make_model

model = make_model()
gen = torch.Generator().manual_seed(1)
prompt_a = torch.randint(0, 512, (1, 32), generator=gen)
prompt_b = torch.randint(0, 512, (1, 32), generator=gen)


@torch.no_grad()
def reference(idx, new=40):
    for _ in range(new):
        idx = torch.cat([idx, model(idx)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return idx[0, -new:].tolist()


@torch.no_grad()
def cached(idx, caches, new=40, keep_position=True):
    start = sum(0 if "k" not in c else c["k"].shape[2] for c in caches[:1])
    logits = model(idx, caches, start=start)
    out = []
    for _ in range(new):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt.item())
        pos = caches[0]["k"].shape[2] if keep_position else 0  # lỗi: quên vị trí
        logits = model(nxt, caches, start=pos)
    return out


def report(name, got, want):
    same = sum(g == w for g, w in zip(got, want))
    first = next((i for i, (g, w) in enumerate(zip(got, want)) if g != w), None)
    print(f"{name:<26} match {same:>2}/40, first diff at step {first}")


ref_b = reference(prompt_b)
report("correct cache", cached(prompt_b, [{} for _ in model.blocks]), ref_b)
report("position reset to 0", cached(prompt_b, [{} for _ in model.blocks], keep_position=False), ref_b)
shared = [{} for _ in model.blocks]
cached(prompt_a, shared)  # yêu cầu của khách A dùng bộ cache này trước
report("cache reused from user A", cached(prompt_b, shared), ref_b)
