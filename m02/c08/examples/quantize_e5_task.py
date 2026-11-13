import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(2)
FAQ = ["Phí giao tiêu chuẩn 18.000 đồng cho 3 km đầu, mỗi km thêm 4.000 đồng.",
       "Hỏa tốc giao trong 2 giờ ở nội thành, cộng thêm 15.000 đồng.",
       "Được hủy miễn phí khi shipper chưa nhận hàng.",
       "Đơn thu tiền hộ tối đa 5.000.000 đồng.",
       "Hàng dễ vỡ hư hỏng được bồi thường tối đa 2.000.000 đồng nếu có ảnh lúc nhận.",
       "Tổng đài làm việc từ 7 giờ đến 22 giờ, kể cả ngày lễ.",
       "Đơn tới Khu Công Nghiệp có thể chậm 30 phút vào giờ tan ca.",
       "Shipper không nhận chuyển khoản vào tài khoản cá nhân."]
QUESTIONS = ["ship 5 km hết bao nhiêu tiền", "muốn giao gấp trong vài tiếng thì sao",
             "shipper chưa lấy hàng, tôi hủy có mất phí không", "COD được tối đa bao nhiêu",
             "ly thủy tinh bị vỡ thì đền thế nào", "mấy giờ gọi tổng đài được",
             "sao giao vào khu công nghiệp lúc 17h chậm vậy", "tôi chuyển khoản cho anh shipper được không"]
model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
original = {n: p.detach().clone() for n, p in model.named_parameters() if p.ndim == 2 and "embeddings" not in n}


def fake_quantize(bits, group=32):
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in original:
                w = original[name].reshape(-1, group)
                scale = w.abs().amax(dim=1, keepdim=True) / qmax
                p.copy_((torch.clamp(torch.round(w / scale), -qmax, qmax) * scale).reshape(p.shape))


def evaluate():
    q = model.encode(["query: " + x for x in QUESTIONS], normalize_embeddings=True, convert_to_tensor=True)
    d = model.encode(["passage: " + x for x in FAQ], normalize_embeddings=True, convert_to_tensor=True)
    top1 = (q @ d.T).argmax(dim=1)
    missed = [i for i in range(len(FAQ)) if top1[i] != i]  # câu hỏi bị ghép sai đoạn FAQ
    return q, len(FAQ) - len(missed), missed


ref, hits, _ = evaluate()
n_weights = sum(w.numel() for w in original.values())
print(f"quantized linear weights: {n_weights / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
print(f"{'weights':<10}{'top-1 hits':>11}{'cos to fp32':>13}  missed questions")
print(f"{'fp32':<10}{hits:>8}/{len(FAQ)}{1.0:>13.4f}  -")
for bits in (8, 4, 3, 2):
    fake_quantize(bits)
    emb, hits, missed = evaluate()
    print(f"{'int' + str(bits):<10}{hits:>8}/{len(FAQ)}{(emb * ref).sum(dim=1).mean():>13.4f}  {missed}")
