import torch
from transformers import AutoConfig, BertModel

torch.manual_seed(0)
config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
config.num_hidden_layers = 2  # giữ 2 lớp cho nhanh; giới hạn vị trí không đổi
model = BertModel(config).eval()
print("position table:", tuple(model.embeddings.position_embeddings.weight.shape))

ids = torch.randint(1000, 2000, (1, 600))  # một cuộc hội thoại dài 600 token
with torch.no_grad():
    out = model(input_ids=ids[:, :512], token_type_ids=torch.zeros_like(ids[:, :512]))
print("512 tokens ok:", tuple(out.last_hidden_state.shape))
with torch.no_grad():
    model(input_ids=ids, token_type_ids=torch.zeros_like(ids))  # vượt 512 vị trí đã học
