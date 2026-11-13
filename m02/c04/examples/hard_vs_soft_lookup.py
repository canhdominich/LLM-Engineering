import numpy as np

# Bảng chính sách: tên, khóa 2 chiều [mức trễ, mức hư hỏng], giá trị = số giờ xử lý cam kết
POLICIES = {"late delivery": 24.0, "damaged goods": 48.0, "cancel and refund": 72.0}
KEYS = np.array([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
VALUES = np.array([24.0, 48.0, 72.0])

print("hard lookup 'late delivery':", POLICIES.get("late delivery"))
print("hard lookup 'delivery is so slow':", POLICIES.get("delivery is so slow"))


def softmax(x):
    e = np.exp(x - x.max())  # trừ max để exp không tràn số
    return e / e.sum()


def soft_lookup(query, keys, values, scale=1.0):
    scores = keys @ query * scale  # mỗi khóa một điểm: tích vô hướng với query
    weights = softmax(scores)      # điểm thành trọng số, tổng bằng 1
    return scores, weights, weights @ values


query = np.array([1.0, 0.5])  # "delivery is so slow": trễ nhiều, hư hỏng ít
for scale in (1.0, 3.0):
    scores, weights, answer = soft_lookup(query, KEYS, VALUES, scale)
    print(f"scale={scale}: scores={scores.round(2)} weights={weights.round(3)} "
          f"sum={weights.sum():.3f} answer={answer:.1f} hours")
