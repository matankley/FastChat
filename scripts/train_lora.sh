deepspeed fastchat/train/train_lora.py \
    --model_name_or_path lmsys/vicuna-7b-v1.3  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path $DATA_PATH \
    --output_dir ./checkpoints \
    --num_train_epochs 150 \
    --fp16 True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100  \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --q_lora False \
    --deepspeed $PATH_TO_DEEPSPEED_CONFIG \
    --gradient_checkpointing True \
    --flash_attn False


python fastchat/train/train_lora.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path hf-Universal-NER/Pile-NER-type \
    --output_dir ./checkpoints \
    --num_train_epochs 1 \
    --bf16 True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 40  \
    --save_strategy "steps" \
    --save_steps 20 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --q_lora True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --flash_attn False



pip install bitsandbytes deepspeed scipy