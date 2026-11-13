import copy
import io
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from _tinygpt import answer, cached, sft

tok, base, _ = cached("pretrain")
questions = ["phí giao nội quận bao nhiêu?", "tổng đài làm việc mấy giờ?", "hủy đơn có mất phí không?"]

full = copy.deepcopy(base)
full_losses = sft(full, tok)

config = LoraConfig(r=4, lora_alpha=8, target_modules=["qkv", "proj", "fc", "out"])
lora = get_peft_model(copy.deepcopy(base), config)
lora.print_trainable_parameters()
# LoRA học ít tham số nên thường cần learning rate cao hơn
lora_losses = sft(lora, tok, lr=3e-3)

buf = io.BytesIO()
torch.save(full.state_dict(), buf)
with tempfile.TemporaryDirectory() as tmp:
    lora.save_pretrained(tmp)
    adapter_bytes = (Path(tmp) / "adapter_model.safetensors").stat().st_size

print(f"final SFT loss: full {full_losses[-1]:.3f}, LoRA {lora_losses[-1]:.3f}")
print(f"saved size: full model {buf.tell() / 1024:.0f} KB, LoRA adapter {adapter_bytes / 1024:.0f} KB")
for q in questions:
    print(f"{q}\n  full: {answer(full, tok, q)}\n  LoRA: {answer(lora, tok, q)}")
