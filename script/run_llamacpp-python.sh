set -e
cuda_devices=0
src_file=lyu/floresp-v2.0-rc.2/devtest/devtest.eng_Latn
ref_file=lyu/floresp-v2.0-rc.2/devtest/devtest.jpn_Jpan

output_prefix=/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-pissa/checkpoint-33/merge
model_path=$output_prefix/model_Q4_K_M.gguf
tgt_file=$output_prefix/devtest.llama.cpp.output

python /home/2/uh02312/lyu/MT/src/inference_llamacpp.py \
    --src_file $src_file \
    --output_file $tgt_file \
    --model_path $model_path \
    --gpu $cuda_devices
bash lyu/MT/src/eval_ja.sh $src_file $ref_file $tgt_file $output_prefix $cuda_devices