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

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from prompt_temple import formatting_prompts_func,end_of_prompt
if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

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


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

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
    model_cache_dir= '/home/2/uh02312/lyu/checkpoints'
    train_file_path=args.dataset_name
    eval_text_path= '/home/2/uh02312/lyu/MT/data/flores200_dev.en-ja.bi.json'
    train_dataset=load_dataset("json", data_files=train_file_path)["train"]
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset=load_dataset("json", data_files=eval_text_path)["train"]
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
    #print random examples
    random_list = random.sample(range(0, len(train_dataset)), 5)
    print("Train dataset example: ")
    for i in random_list:
        print(train_dataset[i]["text"]+"\n")
    random_list = random.sample(range(0, len(eval_dataset)), 5)
    print("Eval dataset example: ")
    for i in random_list:
        print(eval_dataset[i]["text"]+"\n")
    print("Train dataset size: ",len(train_dataset))
    print("Eval dataset size: ",len(eval_dataset))
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, cache_dir=model_cache_dir)
    tokenizer.pad_token = "<|reserved_special_token_250|>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_250|>")
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16, cache_dir=model_cache_dir, **model_kwargs).cuda()

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )
    response_template_ids = tokenizer.encode(end_of_prompt)
    print("end_of_prompt: ",end_of_prompt)
    print("end_of_prompt_ids: ",response_template_ids)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            data_collator=collator,
            dataset_text_field = "text",
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)