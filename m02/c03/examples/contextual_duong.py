import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

NAME = "intfloat/multilingual-e5-small"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME).eval()

REF_ROAD = ["đường bị ngập nên shipper giao trễ", "shipper báo đường Bến Sông đang sửa", "xe không qua được vì đường kẹt cứng"]
REF_SUGAR = ["Tạp Hóa Cô Tấm đặt năm kg đường", "cho tôi một gói đường và một hộp sữa", "đơn thiếu một bao đường cát"]
TEST = [("trời mưa to, đường trơn lắm", "road"), ("con đường vào Xóm Vui rất hẹp", "road"),
        ("tôi ở cuối đường, gần chợ", "road"), ("đường hết rồi, giao thêm hai gói nhé", "sugar"),
        ("trà sữa ít đường, nhiều đá", "sugar"), ("bánh này ngọt quá, chắc nhiều đường", "sugar")]


def duong_vectors(sentences):
    enc = tok(sentences, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hidden = torch.stack(out.hidden_states, dim=1)  # (batch, 13 tầng, seq, 384)
    rows = []
    for b in range(len(sentences)):
        pos = tok.convert_ids_to_tokens(enc["input_ids"][b]).index("▁đường")
        rows.append(hidden[b, :, pos])
    return torch.stack(rows).numpy()  # (batch, 13, 384)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


road, sugar, test = duong_vectors(REF_ROAD), duong_vectors(REF_SUGAR), duong_vectors([s for s, _ in TEST])
print("tokens:", tok.tokenize("đường ngập, giao trễ"))
print("vectors of 'đường' per sentence:", test.shape)
for layer in [0, 3, 6, 9, 12]:
    c_road, c_sugar = unit(unit(road[:, layer]).mean(0)), unit(unit(sugar[:, layer]).mean(0))
    t = unit(test[:, layer])
    margin = t @ c_road - t @ c_sugar  # dương: gần nghĩa "con đường" hơn
    pred = np.where(margin > 0, "road", "sugar")
    sep = np.mean([m if lab == "road" else -m for m, (_, lab) in zip(margin, TEST)])
    print(f"layer {layer:2d}: correct {sum(p == lab for p, (_, lab) in zip(pred, TEST))}/6, separation {sep:+.3f}, hard sentence margin {margin[3]:+.3f}")
    if layer == 12:
        for (sentence, lab), m, p in zip(TEST, margin, pred):
            print(f"   {sentence:38} true={lab:5} pred={p:5} margin={m:+.3f}")
