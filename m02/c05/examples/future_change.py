import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, Qwen3Model

torch.manual_seed(0)
a = "hủy đơn, không đổi địa chỉ"
b = "hủy đơn, không đổi địa điểm"  # chỉ khác từ cuối


def changes(tok, model, text_a, text_b):
    ia = tok(text_a, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ib = tok(text_b, return_tensors="pt", add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        ha, hb = model(input_ids=ia).last_hidden_state[0], model(input_ids=ib).last_hidden_state[0]
    names = [tok.decode([t]) for t in ia[0]]  # in lại chữ của từng token
    return [(names[i], (ha[i] - hb[i]).abs().max().item()) for i in range(ia.shape[1])]


enc_tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
encoder = AutoModel.from_pretrained("intfloat/multilingual-e5-small").eval()  # encoder kiểu BERT, đã huấn luyện

dec_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")  # chỉ tải tokenizer
cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 256, 512
cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 64
decoder = Qwen3Model(cfg).eval()  # kiến trúc Qwen3 thu nhỏ, trọng số ngẫu nhiên

for name, rows in [("encoder (e5-small)", changes(enc_tok, encoder, a, b)), ("decoder (Qwen3 layout)", changes(dec_tok, decoder, a, b))]:
    print(name)
    for i, (t, d) in enumerate(rows):
        print(f"  pos {i}  change {d:.4f}  {t!r}")
