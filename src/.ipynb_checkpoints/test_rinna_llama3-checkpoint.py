import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def create_input(text, tokenizer):
    text=tokenizer.bos_token+text
    input_ids = tokenizer.encode(text, return_tensors="pt")
    return input_ids

model_id = "rinna/llama-3-youko-8b"
model=AutoModelForCausalLM.from_pretrained(model_id,cache_dir="lyu/checkpoints",torch_dtype=torch.bfloat16).cuda()
tokenizer=AutoTokenizer.from_pretrained(model_id,cache_dir="lyu/checkpoints")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

text="今日はいい天気ですね。"
input_ids = create_input(text, tokenizer).to(model.device)
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.6,
    eos_token_id=terminators,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

