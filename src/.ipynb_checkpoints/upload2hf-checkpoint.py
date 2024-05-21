from huggingface_hub import login

login()
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
#モデルのレポジトリ名を定義します。
repo_name = "lyu-boxuan/llama-3-youko-8b-En-Ja-MT-LoRA"
model_id = "lyu/MT/output/llama3-sft-lora-16-NLLB-10k-run2/checkpoint-132/merge"
model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").cuda()
tokenizer=AutoTokenizer.from_pretrained(model_id, use_fast=True)
#トークナイザーをアップロードします
tokenizer.push_to_hub(repo_name)

#モデルをアップロードします
model.push_to_hub(repo_name)