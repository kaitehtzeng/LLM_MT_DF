import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_name = "/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-pissa/checkpoint-33"   #学習済みadapter_config.jsonのパス指定
output_dir = "/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-pissa/checkpoint-33/merge"  #マージモデルの出力先

# PEFT(LoRA)の指定
peft_config = PeftConfig.from_pretrained(peft_name)
# ベースモデルの読み込み
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path,use_fast=True)
tokenizer.pad_token = "<|reserved_special_token_250|>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_250|>")
# PEFT(LoRA)の読み込み
model = PeftModel.from_pretrained(model, peft_name)
model.generation_config.pad_token_id = tokenizer.pad_token_id
#set as beam search
model.generation_config.do_sample = False
model.generation_config.num_beams = 5
model.generation_config.eary_stopping = True
# マージモデル作成
merged_model = model.merge_and_unload()
# 出力
merged_model.save_pretrained(output_dir)  
tokenizer.save_pretrained(output_dir)
print(f"Saving to {output_dir}")  