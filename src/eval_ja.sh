#!/bin/bash
set -e
# 接收参数
src_file=$1       # 源文件
tgt_file=$2       # 目标文件
hypo_file=$3      # 预测文件
output_prefix=$4  # 输出前缀
cuda_devices=$5   # 使用的GPU
lang=$6           # 语言
# 现有的命令，使用新的变量
sacrebleu $tgt_file -i $hypo_file -m bleu chrf -tok ja-mecab --confidence > $output_prefix/score.log
CUDA_VISIBLE_DEVICES=$cuda_devices comet-score -s $src_file -t $hypo_file -r $tgt_file  --quiet --only_system --model_storage_path lyu/cache >> $output_prefix/score.log
CUDA_VISIBLE_DEVICES=$cuda_devices /gs/bs/tga-t2glrlab-1/lyu/miniconda3/envs/tf/bin/python -m bleurt.score_files \
  -candidate_file=$hypo_file \
  -reference_file=$tgt_file \
  -bleurt_batch_size=100 \
  -batch_same_length=True \
  -bleurt_checkpoint=/gs/bs/tga-t2glrlab-1/lyu/bleurt/BLEURT-20 > $output_prefix/bleurt.score

# 平均分数
awk '{sum+=$1} END {print sum/NR}' $output_prefix/bleurt.score >> $output_prefix/score.log