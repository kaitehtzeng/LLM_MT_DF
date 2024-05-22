import os
import argparse
parser = argparse.ArgumentParser(description='Translate English to Japanese')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
parser.add_argument('--gpu', type=str, default="0", help='GPU number to use, e.g., "0"')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--beam_size', type=int, default=5)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from transformers import AutoTokenizer
import ctranslate2
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

beam_size = args.beam_size
response_template= "\n###  日本語：\n"
prefix="###  次の英語のテキストを日本語に翻訳してください：\n英語：\n"
def create_input(texts, tokenizer):
    token_list = []
    for src in texts:
        fromatted_texts=f"{prefix}{src}{response_template}"
        fromatted_texts = tokenizer.encode(fromatted_texts)
        token_list.append(tokenizer.convert_ids_to_tokens(fromatted_texts))
    return token_list

def generate_batch(src_lines, tokenizer, model, prefix_tokens):
    input_tokens = create_input(src_lines, tokenizer)
    outputs = model.generate_batch(
        input_tokens,
        beam_size=beam_size,
        include_prompt_in_result=False,
        static_prompt=prefix_tokens)
    mt=[]
    for output in outputs:
        mt.append(output.sequences_ids[0])
    outputs = tokenizer.batch_decode(mt, skip_special_tokens=True)
    return outputs
src_file = args.src_file
output_file = args.output_file
batch_size= args.batch_size
model_id = args.model_path
tokenizer_path = args.tokenizer_path
model=ctranslate2.Generator(model_id, device="cuda")
tokenizer=AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, cache_dir="/home/2/uh02312/lyu/checkpoints")
tokenizer.pad_token = "<|reserved_special_token_250|>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_250|>")
prefix_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prefix))
with open(src_file, "r", encoding="utf-8") as f:
    src_lines = [line.strip() for line in f.readlines()]
    print("src_lines", len(src_lines))

results=[]
for i in tqdm(range(0, len(src_lines), batch_size)):
    batch_mt=generate_batch(src_lines[i:i+batch_size], tokenizer, model,prefix_tokens)
    results.extend(batch_mt)

with open(output_file, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line.replace("\n", "")+"\n")