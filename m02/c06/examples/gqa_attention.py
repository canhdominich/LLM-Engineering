import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, D, N_Q, N_KV = 1, 6, 64, 8, 2
head_dim = D // N_Q
x = torch.randn(B, T, D)
w_q = torch.nn.Linear(D, N_Q * head_dim, bias=False)
w_k = torch.nn.Linear(D, N_KV * head_dim, bias=False)
w_v = torch.nn.Linear(D, N_KV * head_dim, bias=False)

q = w_q(x).view(B, T, N_Q, head_dim).transpose(1, 2)
k = w_k(x).view(B, T, N_KV, head_dim).transpose(1, 2)
v = w_v(x).view(B, T, N_KV, head_dim).transpose(1, 2)
print("q", tuple(q.shape), "| k", tuple(k.shape), "| v", tuple(v.shape))

# mỗi nhóm 4 query head dùng chung một cặp K, V
group = N_Q // N_KV
k_rep, v_rep = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
manual = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
builtin = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
print("k after repeat", tuple(k_rep.shape), "| output", tuple(manual.shape))
print("max diff manual vs enable_gqa:", f"{(manual - builtin).abs().max().item():.1e}")
print("query heads 0-3 share kv head 0:", torch.equal(k_rep[:, 0], k_rep[:, 3]), "| head 4 uses kv head 1:", torch.equal(k_rep[:, 4], k[:, 1]))
mha_kv = 2 * D * D
gqa_kv = w_k.weight.numel() + w_v.weight.numel()
print(f"K,V projection params: MHA {mha_kv} | GQA {gqa_kv} ({gqa_kv / mha_kv:.0%})")
