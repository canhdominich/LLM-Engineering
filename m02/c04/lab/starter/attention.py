import numpy as np


def softmax(x, axis=-1):
    # TODO 1: softmax ổn định số: trừ giá trị lớn nhất theo trục axis trước khi exp.
    # Hàng toàn -inf (bị che hết) phải trả về toàn số 0, không được ra nan.
    raise NotImplementedError


def causal_mask(n):
    # TODO 2: trả về mảng bool shape (n, n), True ở các ô được phép nhìn (cột j <= hàng i).
    raise NotImplementedError


def scaled_dot_product_attention(q, k, v, mask=None, is_causal=False):
    # TODO 3: q (..., n, d), k (..., m, d), v (..., m, dv), trả về (..., n, dv).
    # Báo ValueError khi d của q và k khác nhau hoặc số key khác số value.
    # Điểm = q @ k chuyển vị / căn bậc hai của d. mask là bool, True = được phép nhìn (giống torch).
    # is_causal=True thì thêm causal_mask. Ô không được phép nhìn đặt -inf trước softmax.
    raise NotImplementedError


def split_heads(x, n_heads):
    # TODO 4: (n, d_model) thành (n_heads, n, d_model // n_heads).
    # Báo ValueError nếu d_model không chia hết cho n_heads.
    raise NotImplementedError


def merge_heads(x):
    # TODO 5: (n_heads, n, d_head) thành (n, n_heads * d_head), ngược với split_heads.
    raise NotImplementedError


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=False):
    # TODO 6: chiếu x bằng Wq, Wk, Wv (mỗi ma trận (d_model, d_model)), tách đầu,
    # chạy scaled_dot_product_attention trên mọi đầu cùng lúc, ghép đầu, nhân Wo.
    raise NotImplementedError


def future_leak(x, Wq, Wk, Wv, Wo, n_heads, causal):
    # TODO 7: cộng 10 vào token cuối của một bản sao x, chạy multi_head_attention trên x và bản sao,
    # trả về float: chênh lệch tuyệt đối lớn nhất ở mọi vị trí trừ vị trí cuối.
    raise NotImplementedError


if __name__ == "__main__":
    import torch

    rng = np.random.default_rng(0)
    n, d_model, n_heads = 6, 16, 4
    x = rng.normal(size=(n, d_model))
    Wq, Wk, Wv, Wo = (rng.normal(size=(d_model, d_model)) / 4 for _ in range(4))
    out = multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=True)
    mha = torch.nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.from_numpy(np.concatenate([Wq.T, Wk.T, Wv.T])))
        mha.out_proj.weight.copy_(torch.from_numpy(Wo.T))
        t = torch.from_numpy(x)[None]
        ref, _ = mha(t, t, t, attn_mask=torch.nn.Transformer.generate_square_subsequent_mask(n, dtype=torch.float64), need_weights=False)
    print("output shape:", out.shape)
    print("max abs diff vs torch MultiheadAttention (causal):", f"{np.abs(out - ref[0].numpy()).max():.1e}")
    print("future leak without mask:", round(future_leak(x, Wq, Wk, Wv, Wo, n_heads, causal=False), 3))
    print("future leak with causal mask:", future_leak(x, Wq, Wk, Wv, Wo, n_heads, causal=True))
    print("softmax of a fully masked row:", softmax(np.array([-np.inf, -np.inf, -np.inf])))
