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
from torchtext.data.metrics import bleu_score
TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)
from transformers import TrainingArguments,TrainerCallback
from   transformers.integrations import WandbCallback
from unsloth import is_bfloat16_supported
from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from prompt_temple import formatting_prompts_func,end_of_prompt,formatting_prompts_func_eval
if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler
from comet import download_model, load_from_checkpoint
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import corpus_bleu
import torch
from datasets import load_dataset, disable_caching
from evaluate import load
import evaluate
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
    wandb.init(project="tryout")
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    dtype = torch.float16
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    wandb.config.update(vars(args))
    wandb.config.update(vars(training_args))
    wandb.config.update(vars(model_config))
    wandb.config.update({'dtype':torch.float16,'max_seq_length':2048,'load_in_4bit':True},allow_val_change=True)
    generate_params = {
        "max_new_tokens": 256,
        "num_beams": 5,
        "early_stopping": True,
        "do_sample": False,
    }
    wandb.config.update(generate_params)
    
    
    
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
    eval_dataset=load_dataset("json", data_files=eval_text_path)["train"]
    eval_dataset_tr = eval_dataset.map(formatting_prompts_func,batched=True)
    #print random examples
    random_list = random.sample(range(0, len(train_dataset)), 5)
    print("Train dataset example: ")
    for i in random_list:
        print(train_dataset[i]["text"]+"\n")
    random_list = random.sample(range(0, len(eval_dataset_tr)), 5)
    print("Eval dataset example: ")
    for i in random_list:
        print(eval_dataset_tr[i]["text"]+"\n")
    print("Train dataset size: ",len(train_dataset))
    print("Eval dataset size: ",len(eval_dataset_tr))
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
    from evaluate import load
    bleau = evaluate.load('bleu')
    comet_metric = load('comet')
    def preprocess_logits_for_metrics(logits,labels):
        if isinstance(logits,tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    def compute_metrics(eval_preds):
        preds,labels = eval_preds
        labels= labels[:,1:]
        preds = preds[:,:-1]
        labels[mask]= tokenizer
        mask = labels == -100

        labels[mask]=tokenizer.pad_token_id
        preds[mask]=tokenizer.pad_token_id

        d_labels = tokenizer.batch_decode(labels,skip_special_tokens=True)
        d_preds = tokenizer.batch_decode(preds,skip_special_tokens=True)
        bleu_score = bleu.compute(predictions=d_preds, references= d_labels)
        comet_score = comet_metric.compute(predictions = d_preds,references = d_labels)

        return {**bleu_score,**comet_score}



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
    compute_metrics=compute_metrics
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
    repo_id="kaitehtzeng/llma",
    use_auth_token=True
    )

