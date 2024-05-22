set -e
cuda_devices=0
src_file=lyu/floresp-v2.0-rc.2/devtest/devtest.eng_Latn
ref_file=lyu/floresp-v2.0-rc.2/devtest/devtest.jpn_Jpan
tokenizer_path=/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-pissa/checkpoint-33/merge
model_path=/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-pissa/checkpoint-33/merge/ct2
output_prefix=/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-pissa/checkpoint-33/merge/ct2/gs
tgt_file=$output_prefix/devtest.output

python lyu/MT/src/inference_ct2.py \
    --src_file $src_file \
    --output_file $tgt_file \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --gpu $cuda_devices \
    --batch_size 32 \
    --beam_size 1

bash lyu/MT/src/eval_ja.sh $src_file $ref_file $tgt_file $output_prefix $cuda_devices