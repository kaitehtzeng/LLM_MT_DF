set -e 
cuda_devices=0
run_name=llama3-sft-lora-wmt_20-23_en-ja-bi-lion-pissa
python ./src/llama3_trl_sft.py \
    --model_name_or_path rinna/llama-3-youko-8b \
    --dataset_name ./data/wmt_20-23.en-ja.bi.json \
    --learning_rate 2e-4 \
    --max_seq_length 4096 \
    --output_dir ./output/$run_name \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing \
    --logging_dir ./output/logs/$run_name \
    --prediction_loss_only \
    --load_best_model_at_end \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=10 \
    --save_total_limit=1 \
    --weight_decay=0.01 \
    --warmup_steps=5 \
    --lr_scheduler_type="linear" \
    --evaluation_strategy="steps" \
    --optim="lion_8bit" \
    --save_steps=1 \
    --logging_steps 1 \
    --group_by_length True \
    --use_peft \
    --lora_r=64 \
    --init_lora_weights="pissa" \
    --lora_alpha=16 \
    --fp16=1 \
    --bf16=0 \
    --max_steps= 50\
    --seed 3407\

