from huggingface_hub import login

login()
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
#モデルのレポジトリ名を定義します。
repo_name = "lyu-boxuan/llama-3-youko-8b-En-Ja-MT-LoRA"
model_id = "/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-pissa/checkpoint-33/merge"
model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").cuda()
tokenizer=AutoTokenizer.from_pretrained(model_id, use_fast=True)
#トークナイザーをアップロードします
tokenizer.push_to_hub(repo_name)

#モデルをアップロードします
model.push_to_hub(repo_name)