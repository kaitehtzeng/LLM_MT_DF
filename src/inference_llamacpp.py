import os
import argparse
parser = argparse.ArgumentParser(description='Translate English to Japanese')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--gpu', type=str, default="0", help='GPU number to use, e.g., "0"')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from tqdm import tqdm
from llama_cpp import Llama

response_template= "\n###  日本語：\n"
prefix="###  次の英語のテキストを日本語に翻訳してください：\n英語：\n"
def create_input(text):
    fromatted_text=f"{prefix}{text}{response_template}"
    return fromatted_text

def generate(input_text, model):
    input_text = create_input(input_text)
    output = model(input_text, max_tokens=256, echo=False,top_k=1, temperature=0.0)
    output = output['choices'][0]['text']
    return output.replace("\n", "")
src_file = args.src_file
output_file = args.output_file
model_id = args.model_path
llm = Llama(
      model_path=model_id,
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      seed=1337, # Uncomment to set a specific seed
      n_ctx=2048, # Uncomment to increase the context window
)

with open(src_file, "r", encoding="utf-8") as f:
    src_lines = [line.strip() for line in f.readlines()]
    print("src_lines", len(src_lines))

results=[]
for i in tqdm(range(len(src_lines))):
    src = src_lines[i]
    output = generate(src, llm)
    results.append(output)
with open(output_file, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line+"\n")