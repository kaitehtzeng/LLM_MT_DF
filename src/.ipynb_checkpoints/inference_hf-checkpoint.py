import os
import argparse
parser = argparse.ArgumentParser(description='Translate English to Japanese')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--gpu', type=str, default="0", help='GPU number to use, e.g., "0"')
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

response_template= "\n###  日本語：\n"
prefix="###  次の英語のテキストを日本語に翻訳してください：\n英語:\n"
def create_input(texts, tokenizer):
    fromatted_texts = []
    for src in texts:
        fromatted_texts.append(f"{prefix}{src}{response_template}")
    input_ids = tokenizer(fromatted_texts, return_tensors="pt", padding=True)
    return input_ids

def generate_batch(src_lines, tokenizer, model, batch_size=16):
    input_ids = create_input(src_lines, tokenizer).to(model.device)
    outputs = model.generate(
        **input_ids,
        max_new_tokens=256,
        num_beams=5,
        early_stopping=True,
        do_sample=False
    )
    mt=[]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    input_texts = tokenizer.batch_decode(input_ids["input_ids"], skip_special_tokens=True)
    for input_text, mt_text in zip(input_texts, outputs):
        mt.append(mt_text[len(input_text):].replace("\n", ""))
    return mt
src_file = args.src_file
output_file = args.output_file
batch_size= args.batch_size
model_id = args.model_path
model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").cuda()
tokenizer=AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = "<|reserved_special_token_250|>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_250|>")

with open(src_file, "r", encoding="utf-8") as f:
    src_lines = [line.strip() for line in f.readlines()]
    print("src_lines", len(src_lines))

results=[]
for i in tqdm(range(0, len(src_lines), batch_size)):
    batch_mt=generate_batch(src_lines[i:i+batch_size], tokenizer, model, batch_size)
    results.extend(batch_mt)

with open(output_file, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line+"\n")