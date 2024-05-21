#获取目录下所有以checkpoint开头的文件夹，
#然后逐个进行检查点平均
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
#首先获取目录下所有以checkpoint开头的文件夹的名字
checkpoint_dir="/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-run2"
checkpoint_dirs=[os.path.join(checkpoint_dir,dir) for dir in os.listdir(checkpoint_dir) if dir.startswith("checkpoint")]
print("checkpoint_dirs",checkpoint_dirs)
num_checkpoints=len(checkpoint_dirs)
#然后逐个加载模型，计算平均值
#首先加载第一个模型,作为初始模型，将其权重除以总数
#然后把其他模型的权重除以总数加到初始模型上

model=AutoModelForCausalLM.from_pretrained(checkpoint_dirs[0],torch_dtype=torch.float32)
tokenizer=AutoTokenizer.from_pretrained(checkpoint_dirs[0])
 
for param in model.parameters():
    param.data/=num_checkpoints

for checkpoint_dir in tqdm(checkpoint_dirs[1:],desc="checkpoint averaging",unit="checkpoint",total=num_checkpoints-1):
    model_checkpoint=AutoModelForCausalLM.from_pretrained(checkpoint_dir,torch_dtype=torch.float32)
    for param,checkpoint_param in zip(model.parameters(),model_checkpoint.parameters()):
        param.data+=checkpoint_param.data/num_checkpoints

#保存平均模型
model.save_pretrained("/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-run2/average_model")
tokenizer.save_pretrained("/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-run2/average_model")