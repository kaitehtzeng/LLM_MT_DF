import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

response_template = "\n###  日本語：\n"
prefix = "###  次の英語の文書を日本語に翻訳してください：\n"


def create_input(text, tokenizer):
    text = f"{prefix}{text}{response_template}"
    input_ids = tokenizer.encode(text, return_tensors="pt")
    return input_ids


model_id = "lyu/MT/output/llama3-sft-lora-16-NLLB-100k-run2/merge"
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

en = "LLMs Are Here but Not Quite There Yet"
input_ids = create_input(en, tokenizer).to(model.device)
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    num_beams=5,
    do_sample=False,
    early_stopping=True,
)
response = outputs[0][input_ids.shape[-1] :]
print(tokenizer.decode(response, skip_special_tokens=True))