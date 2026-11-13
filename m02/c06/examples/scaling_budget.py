E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def chinchilla_loss(n_params, n_tokens):
    # công thức khớp số liệu trong bài báo Chinchilla, chỉ đúng với dữ liệu và tokenizer của họ
    return E + A / n_params**ALPHA + B / n_tokens**BETA


budget = 6 * 1.72e9 * 34.4e9
print(f"compute budget C = {budget:.2e} FLOPs (C = 6 x N x D)")
print(f"{'params N':>9} {'tokens D':>9} {'tokens/param':>12} {'loss':>6}")
for n in (0.2e9, 0.5e9, 1.72e9, 5e9, 20e9):
    d = budget / (6 * n)
    print(f"{n / 1e9:>8.2f}B {d / 1e9:>8.1f}B {d / n:>12.1f} {chinchilla_loss(n, d):>6.3f}")

n_qwen, d_qwen = 1.72e9, 36e12
print(f"qwen3 1.7B: 20 x N = {20 * n_qwen / 1e9:.1f}B tokens, trained on {d_qwen / 1e12:.0f}T = {d_qwen / (20 * n_qwen):,.0f}x more")

n_kieu, unique, seen = 212_608, 93_783, 1500 * 32 * 64
print(f"kieu GPT: N = {n_kieu:,}, 20 x N = {20 * n_kieu:,} tokens, unique train chars {unique:,} ({unique / (20 * n_kieu):.1%})")
print(f"kieu GPT: tokens seen {seen:,} = {seen / unique:.1f} passes over the same text, C = {6 * n_kieu * seen:.1e} FLOPs")
