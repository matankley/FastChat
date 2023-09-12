torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train_lora.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --data_path hf-Universal-NER/Pile-NER-type \
    --bf16 False \
    --output_dir output_uniner \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --q_lora True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --lazy_preprocess True \
    --flash_attn False \
    --split_eval 0.005


deepspeed fastchat/train/train_lora.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path hf-Universal-NER/Pile-NER-type \
    --bf16 False \
    --output_dir ./checkpoints \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 1024 \
    --q_lora True \
    --deepspeed playground/deepspeed_config_s2.json \
    --flash_attn False \
    --lazy_preprocess True


## one gpu

python fastchat/train/train_lora.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path hf-Universal-NER/Pile-NER-type \
    --output_dir ./checkpoints \
    --num_train_epochs 1 \
    --bf16 False \
    --per_device_train_batch_size 2 \
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
    --tf32 False \
    --model_max_length 1024 \
    --q_lora True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --flash_attn False


