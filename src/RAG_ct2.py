import os
import argparse
parser = argparse.ArgumentParser(description='Translate English to Japanese')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
parser.add_argument('--index_path', type=str, required=True, help='Path to the RAG database')
parser.add_argument('--index_src', type=str, required=True, help='Path to the src file of the RAG database')
parser.add_argument('--index_tgt', type=str, required=True, help='Path to the tgt file of the RAG database')
parser.add_argument('--gpu', type=str, default="0", help='GPU number to use, e.g., "0"')
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import ctranslate2
from tqdm import tqdm


FEW_SHOT_TEMPLATE = "英語：\n{}\n日本語：\n{}\n"
response_template= "\n###  日本語：\n"
prefix="###  次の英語のテキストを日本語に翻訳してください：\n英語：\n"
index_path=args.index_path
rag_src_file=args.index_src
rag_tgt_file=args.index_tgt
src_file = args.src_file
output_file = args.output_file
batch_size=args.batch_size
beam_size = args.beam_size
print("Loading index")
index = faiss.read_index(index_path)
print("Index loaded")
print("Loading model")
model_emb = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5',cache_folder="/home/2/uh02312/lyu/checkpoints",trust_remote_code=True)
print("Model loaded")
model_id = args.model_path
tokenizer_path = args.tokenizer_path
model=ctranslate2.Generator(model_id, device="cuda")
tokenizer=AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, cache_dir="/home/2/uh02312/lyu/checkpoints",padding_side='left')
tokenizer=AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, cache_dir="/home/2/uh02312/lyu/checkpoints",padding_side='left')
tokenizer.pad_token = "<|reserved_special_token_250|>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_250|>")
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def get_batch_few_shot_demo(input_sentences, src_sentences, tgt_sentences, shot=5):
    input_emb=model_emb.encode(input_sentences, show_progress_bar = True)
    _, I = index.search(input_emb, shot)
    demo_src, demo_tgt = [], []
    for i in range(len(input_sentences)):
        src = []
        tgt = []
        for j in range(shot):
            src.append(src_sentences[I[i][j]])
            tgt.append(tgt_sentences[I[i][j]])
        demo_src.append(src)
        demo_tgt.append(tgt)
    return demo_src, demo_tgt


def get_batch_few_shot_prompt(input_sentences, src_lines, tgt_lines, shot=5):
    formatted_prompts = []
    demo_src, demo_tgt = get_batch_few_shot_demo(input_sentences, src_lines, tgt_lines, shot)
    for i in range(len(input_sentences)):
        prompt = []
        for j in range(shot):
            prompt.append(FEW_SHOT_TEMPLATE.format(demo_src[i][j], demo_tgt[i][j]))
        formatted_prompts.append("\n".join(prompt)+ "\n" + prefix + input_sentences[i] + response_template)
    return formatted_prompts
    
    
def generate_translation(input_sentences, tokenizer, model):
    input_tokens = []
    for src in input_sentences:
        src = tokenizer.encode(src)
        input_tokens.append(tokenizer.convert_ids_to_tokens(src))
    outputs = model.generate_batch(
        input_tokens,
        beam_size=beam_size,
        include_prompt_in_result=False)
    mt=[]
    for output in outputs:
        mt.append(output.sequences_ids[0])
    outputs = tokenizer.batch_decode(mt, skip_special_tokens=True)
    return outputs
rag_src_lines = read_file(rag_src_file)
rag_tgt_lines = read_file(rag_tgt_file)
src_lines = read_file(src_file)
assert len(rag_src_lines) == len(rag_tgt_lines)

print("Creating RAG prompt")
formatted_prompts = get_batch_few_shot_prompt(src_lines, rag_src_lines, rag_tgt_lines, 5)
print("Prompt created")
results=[]
for i in tqdm(range(0, len(formatted_prompts), batch_size)):
    batch_mt=generate_translation(formatted_prompts[i:i+batch_size], tokenizer, model)
    results.extend(batch_mt)

with open(output_file, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line+"\n")

