set -e
cuda_devices=0
src_file=lyu/floresp-v2.0-rc.2/devtest/devtest.eng_Latn
ref_file=lyu/floresp-v2.0-rc.2/devtest/devtest.jpn_Jpan

output_prefix=lyu/MT/output/llama3-sft-lora-NLLB_COMET-run2/checkpoint-49
tgt_file=$output_prefix/devtest.output

python lyu/MT/src/inference_hf.py \
    --src_file $src_file \
    --output_file $tgt_file \
    --model_path $output_prefix \
    --gpu $cuda_devices \
    --batch_size 16
    
bash lyu/MT/src/eval_ja.sh $src_file $ref_file $tgt_file $output_prefix $cuda_devices

output_prefix=lyu/MT/output/llama3-sft-lora-wmt20-23-bi-run2/checkpoint-49
tgt_file=$output_prefix/devtest.output

python lyu/MT/src/inference_hf.py \
    --src_file $src_file \
    --output_file $tgt_file \
    --model_path $output_prefix \
    --gpu $cuda_devices \
    --batch_size 16
    
bash lyu/MT/src/eval_ja.sh $src_file $ref_file $tgt_file $output_prefix $cuda_devices