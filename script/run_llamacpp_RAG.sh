set -e
cuda_devices=0
src_file=lyu/floresp-v2.0-rc.2/devtest/devtest.eng_Latn
ref_file=lyu/floresp-v2.0-rc.2/devtest/devtest.jpn_Jpan

output_prefix=/home/2/uh02312/lyu/MT/output/llama3-sft-lora-NLLB_COMET_10k-cometkiwi-da-xl-lion-pissa/checkpoint-33/merge
model_path=$output_prefix/model_Q4_K_M.gguf
output_prefix=$output_prefix/llamacpp/RAG-gs
mkdir -p $output_prefix
tgt_file=$output_prefix/devtest.output
python /home/2/uh02312/lyu/MT/src/RAG_llamacpp.py \
    --src_file $src_file \
    --output_file $tgt_file \
    --model_path $model_path \
    --index_path /home/2/uh02312/lyu/MT/data/RAG/wmt20-23_en.idx \
    --index_src /home/2/uh02312/lyu/MT/data/wmt_20-23.en \
    --index_tgt /home/2/uh02312/lyu/MT/data/wmt_20-23.ja \
    --gpu $cuda_devices \
    --batch_size 64

bash lyu/MT/src/eval_ja.sh $src_file $ref_file $tgt_file $output_prefix $cuda_devices
