set -e
cuda_devices=0
src_file=lyu/floresp-v2.0-rc.2/devtest/devtest.eng_Latn
ref_file=lyu/floresp-v2.0-rc.2/devtest/devtest.jpn_Jpan

output_prefix=/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-pissa/checkpoint-33/merge
tgt_file=$output_prefix/devtest.output

python lyu/MT/src/inference_hf.py \
    --src_file $src_file \
    --output_file $tgt_file \
    --model_path $output_prefix \
    --gpu $cuda_devices \
    --batch_size 16 \
    --beam_size 1

bash lyu/MT/src/eval_ja.sh $src_file $ref_file $tgt_file $output_prefix $cuda_devices