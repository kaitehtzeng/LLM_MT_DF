# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

import logging
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from prompt_temple import formatting_prompts_func,end_of_prompt,formatting_prompts_func_eval
if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler
from comet import download_model, load_from_checkpoint
from nltk.translate.bleu_score import corpus_bleu
import torch
from datasets import load_dataset, disable_caching
disable_caching()

from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from unsloth import FastLanguageModel

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    DataCollatorForCompletionOnlyLM,
)
tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

import wandb

if __name__ == "__main__":
    wand.init(project="llma3 tiny-fine-tune")
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    dtype = torch.float16
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    wandb.config.update(vars(args))
    wandb.config.update(vars(training_args))
    wandb.config.update(vars(model_config))
    wandb.config.update({'dtype':torch.float16,'max_seq_length':2048,'load_in_4bit':True})
    print(training_args)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    ################
    # Dataset
    ################
    model_id= 'rinna/llama-3-youko-8b'
    train_file_path=args.dataset_name
    eval_text_path= './data/flores200_dev.en-ja.bi.json'
    train_dataset=load_dataset("json", data_files=train_file_path)["train"]
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset_tr=load_dataset("json", data_files=eval_text_path)["train"]
    eval_target = [item['tgt'] for item in eval_dataset]
    eval_src = [item['src'] for item in eval_dataset]
    eval_dataset = eval_dataset.map(formatting_prompts_func_eval)
    #print random examples
    random_list = random.sample(range(0, len(train_dataset)), 5)
    print("Train dataset example: ")
    for i in random_list:
        print(train_dataset[i]["text"]+"\n")
    random_list = random.sample(range(0, len(eval_dataset)), 5)
    print("Eval dataset example: ")
    for i in random_list:
        print(eval_dataset_tr[i]["text"]+"\n")
    print("Train dataset size: ",len(train_dataset))
    print("Eval dataset size: ",len(eval_dataset))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    tokenizer.pad_token = "<|reserved_special_token_250|>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_250|>")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    ################
    #Load Comet
    ################
    comet_model_path = download_model("Unbabel/wmt20-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)

    ################
    #Evaluation
    ################
    def generate_batch(text,model,tokenizer):
        input_id = tokenizer(text,return_tensors= pt, padding =True).to(model.device)
        output = model.generate(
        **input,
        max_new_tokens=256,
        num_beams=5,
        early_stopping=True,
        do_sample=False)
        mount = []
        for input,output_id in zip(input_id['input_ids'],output):
            mount.append(output_id[input.size(0):])
        tok = tokenizer.batch_decode(mount, skip_special_tokens=True)
        return tok
    
    def generate_output(dataset,batch_size,model,tokenizer):
        ans = []
        for i in range(0,len(dataset),batch_size):
            batch = generate_batch(data[i:min(len(dataset),i+batch_size)],model,tokenizer)
            ans.extend(batch)
        return ans
    
    
    def evaluate_model(model,tokenizer,eval_dataset,eval_src,eval_target):
        model.eval()
        result = generate_output(eval_dataset,10,model,tokenizer)
        result_corpus = [i.split() for i in result]
        target_corpus = [i.split() for i in eval_target]
        bleu = bleu_score(result_corpus,target_corpus)
        #Calculate comet
        comet_input = [{"src":item['src'],"mt":res,"ref":ref}for item,res,ref in zip(eval_src,results,eval_tag)]
        comet_score = comet_model.predict(comet_input,batch_size=8,gpus=1)
        average_comet_score = sum(comet_score)/len(comet_score)
        return bleu,average_comet_score,result
    
    ################
    # Custom callback for WandB logging
    ###############
    class WandbCallback(TrainerCallback):
        def __init__(self,eval_steps):
            self.eval_steps=eval_steps
            self.step_count = 0
        def on_log(self,args,state,control,logs=None,**kwargs):
            wandb.log(logs)
        
        def on_step_end(self,args,state,control,**kwargs):
            self.step_count+=1
            if self.step_count%self.eval_steps==0:
                bleu_score,comet_score,results = evaluate_model(model,tokenizer,eval_dataset)
                wandb.log({'step':state.global_step,
                           'bleu_score':bleu_score,
                           'comet_score':comet_score,
                           'examples':wandb.Table(columnS=['Input','Target','Prediction']
                                       data=[[item['src'],ref,res] for item,res,ref in zip(eval_src,results,eval_tag)]})


    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )
    response_template_ids = tokenizer.encode("\n######\n")[1:]
    print("end_of_prompt_ids: ",response_template_ids)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    data_collator=collator,
    eval_dataset=eval_dataset_tr,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_args,
    callbacks=[WandbCallback(10)]
    )

    trainer.train()
    wandb.finish()
    with save_context:
        trainer.save_model(training_args.output_dir)

    from huggingface_hub import HfApi
    api=HfApi()
    api = HfApi()
api.upload_folder(
    folder_path="training_args.output_dir",
    repo_id="kaitehtzeng/llma_tryout",
    use_auth_token=True
)

