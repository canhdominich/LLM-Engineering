import copy

import torch
import torch.nn.functional as F

from _anhti_text import PREF_PAIRS, SFT_PAIRS
from _tinygpt import answer, answer_logprob, cached, sft_example

QUESTIONS = ["đơn của tôi đâu rồi?", "phí giao nội quận bao nhiêu?", "hàng của tôi tới chưa?"]


def run(lr, nll_weight, anchor, steps=60, beta=0.1):
    tok, policy, _ = cached("sft")
    reference = copy.deepcopy(policy).requires_grad_(False)
    keep = [sft_example(tok, q, a, policy.cfg.block_size) for q, a in SFT_PAIRS if a != "Không rõ."]
    x = torch.tensor([r[0] for r in keep])
    y = torch.tensor([r[1] for r in keep])
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    for step in range(steps + 1):
        terms = []
        for q, chosen, rejected in PREF_PAIRS:
            lp_c, lp_r = answer_logprob(policy, tok, q, chosen), answer_logprob(policy, tok, q, rejected)
            # margin: policy tăng câu được chọn nhiều hơn câu bị loại bao nhiêu, so với reference
            margin = (lp_c - answer_logprob(reference, tok, q, chosen)) - (lp_r - answer_logprob(reference, tok, q, rejected))
            terms.append(-F.logsigmoid(beta * margin) + nll_weight * (-lp_c / (len(chosen) + 1)))
        dpo = torch.stack(terms).mean()
        loss = dpo + (F.cross_entropy(policy(x).reshape(-1, tok.vocab_size), y.reshape(-1), ignore_index=-100) if anchor else 0)
        if step in (0, steps):
            print(f"  step {step:2d}  loss {loss.item():.4f}  log p(chosen) {lp_c.item():8.2f}  log p(rejected) {lp_r.item():6.2f}")
        if step < steps:
            opt.zero_grad()
            loss.backward()
            opt.step()
    for q in QUESTIONS:
        print(f"  {q} | {answer(policy, tok, q)}")


print("A. DPO only, lr 1e-4")
run(lr=1e-4, nll_weight=0.0, anchor=False)
print("B. DPO + 0.2 x NLL(chosen) + SFT anchor, lr 1e-3")
run(lr=1e-3, nll_weight=0.2, anchor=True)
