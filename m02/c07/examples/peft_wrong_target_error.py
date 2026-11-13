from peft import LoraConfig, get_peft_model

from _tinygpt import setup

tok, model = setup()
print([name for name, _ in model.named_modules()][:8])
# tên q_proj, v_proj là của Llama hay Qwen, không có trong TinyGPT
config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
get_peft_model(model, config)
