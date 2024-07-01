import argparse
parser = argparse.ArgumentParser(description='Translate English to Japanese')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--gpu', type=str, default="0", help='GPU number to use, e.g., "0"')
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=5)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import os

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose
from prompt_temple import formatting_prompts_func,formatting_prompts_func_eval

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler
from datasets import load_dataset, disable_caching
disable_caching()

from tqdm.rich import tqdm
from unsloth import FastLanguageModel

tqdm.pandas()
src_file = args.src_file
output_file = args.output_file
batch_size= args.batch_size
beam_size = args.beam_size
max_seq_length = args.max_seq_length


eval_dataset=load_dataset("json", data_files=src_file)
eval_dataset= eval_dataset.map(formatting_prompts_func,batched=True)

eval_target = [item['tgt'] for item in eval_dataset]
eval_src = [item['src'] for item in eval_dataset]
eval_dataset = eval_dataset.map(formatting_prompts_func_eval)

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "/content/gdrive/MyDrive/output_first/llama3-sft-lora-wmt_20-23_en-ja-bi-adamw-pissa"
        max_seq_length = 1024,
        dtype = torch.float16,
        load_in_4bit = True
)

FastLanguageModel.for_inference(model)

tokenizer.pad_token = "<|reserved_special_token_250|>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_250|>")
model.generation_config.pad_token_id = tokenizer.pad_token_id

def generate_batch(src_lines, tokenizer, model):
    input_ids = tokenizer(src_lines, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        **input_ids,
        max_new_tokens=256,
        num_beams=beam_size,
        early_stopping=True,
        do_sample=False
    )
    mt=[]
    for input_id,output_id in zip(input_ids["input_ids"], outputs):
        mt.append(output_id[input_id.size(0):])
    mt = tokenizer.batch_decode(mt, skip_special_tokens=True)
    return mt
results=[]
for i in tqdm(range(0, len(eval_dataset), batch_size)):
    batch_mt=generate_batch(eval_dataset[i:i+batch_size], tokenizer, model)
    results.extend(batch_mt)

with open(output_file, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line.replace("\n", "") + "\n")

