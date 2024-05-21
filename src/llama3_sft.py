import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm
import json
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Prepare and tokenize dataset
run_name= 'test-1'
model_id= 'rinna/llama-3-youko-8b'
model_cache_dir= 'lyu/checkpoints'
train_file_path="lyu/MT/data/wmt_20-22.en-ja.json"
eval_text_path="lyu/MT/data/wmt_23.en-ja.json"
response_template= "英語："
prefix="次の英語の文書を日本語に翻訳してください："
tokenizer=AutoTokenizer.from_pretrained(model_id,cache_dir=model_cache_dir)

def preprocess_function(examples):
    src=examples["src"]
    tgt=examples["tgt"]
    text=prefix+src+response_template+tgt
    return text
train_dataset=load_dataset("json", data_files=train_file_path)["train"]
eval_dataset=load_dataset("json", data_files=eval_text_path)["train"]
    
print("Train dataset size: ",len(train_dataset))
print("Eval dataset size: ",len(eval_dataset))
# Load pretrained model and evaluate model after each epoch
model=AutoModelForCausalLM.from_pretrained(model_id,cache_dir=model_cache_dir,torch_dtype=torch.bfloat16).cuda()
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    formatting_func=preprocess_function,
    data_collator=collator,
)
trainer.train()