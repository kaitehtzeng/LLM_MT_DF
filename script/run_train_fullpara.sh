set -e 
cuda_devices=0
src_file=lyu/floresp-v2.0-rc.2/devtest/devtest.eng_Latn
ref_file=lyu/floresp-v2.0-rc.2/devtest/devtest.jpn_Jpan
run_name=llama3-sft-NLLB_COMET_12k-cometkiwi-da-xl-lion
python lyu/MT/src/llama3_trl_sft.py \
    --model_name_or_path rinna/llama-3-youko-8b \
    --dataset_name /home/2/uh02312/lyu/MT/data/NLLB_COMET/12K_cometkiwi-da-xl.en-ja.json \
    --learning_rate 5e-6 \
    --max_seq_length 4096 \
    --output_dir lyu/MT/output/$run_name \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing \
    --logging_dir lyu/MT/output/logs/$run_name \
    --prediction_loss_only \
    --load_best_model_at_end \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=64 \
    --num_train_epochs=10 \
    --save_total_limit=1 \
    --weight_decay=0.01 \
    --warmup_steps=10 \
    --lr_scheduler_type="linear" \
    --evaluation_strategy="steps" \
    --optim="lion_8bit" \
    --save_steps=1 \
    --logging_steps 1 \
    --tf32 True \
    --bf16 True \
    --bf16_full_eval True \
    --group_by_length True

set -e 
cuda_devices=0
src_file=lyu/floresp-v2.0-rc.2/devtest/devtest.eng_Latn
ref_file=lyu/floresp-v2.0-rc.2/devtest/devtest.jpn_Jpan
run_name=llama3-sft-NLLB_COMET_15k-cometkiwi-da-xl-lion
python lyu/MT/src/llama3_trl_sft.py \
    --model_name_or_path rinna/llama-3-youko-8b \
    --dataset_name /home/2/uh02312/lyu/MT/data/NLLB_COMET/15K_cometkiwi-da-xl.en-ja.json \
    --learning_rate 5e-6 \
    --max_seq_length 4096 \
    --output_dir lyu/MT/output/$run_name \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing \
    --logging_dir lyu/MT/output/logs/$run_name \
    --prediction_loss_only \
    --load_best_model_at_end \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=64 \
    --num_train_epochs=10 \
    --save_total_limit=1 \
    --weight_decay=0.01 \
    --warmup_steps=10 \
    --lr_scheduler_type="linear" \
    --evaluation_strategy="steps" \
    --optim="lion_8bit" \
    --save_steps=1 \
    --logging_steps 1 \
    --tf32 True \
    --bf16 True \
    --bf16_full_eval True \
    --group_by_length True