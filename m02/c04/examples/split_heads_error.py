import numpy as np

n, d_model, n_heads = 20, 384, 10  # 384 không chia hết cho 10
Q = np.zeros((n, d_model))
d_head = d_model // n_heads
print("d_head =", d_head, "so heads cover", d_head * n_heads, "of", d_model, "dims")
heads = Q.reshape(n, n_heads, d_head)
