export PYTHONPATH=/data4/hzp/ECR-Chain

gpu_id=$1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --include localhost:${gpu_id} --master_port ${MASTER_PORT} EmoReasonLlama/run/train_lora.py \
    --deepspeed /data4/hzp/ECR-Chain/EmoReasonLlama/ds_config_zero2_offload.json \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj", "k_proj", "v_proj", "down_proj", "gate_proj", "up_proj" \
    --model_name_or_path /data4/hzp/pretrained_models/vicuna/vicuna-7b-v1.3/ \
    --raw_data_path /data4/hzp/ECR-Chain/RECCON_all_data.json \
    --conv_templ_type "vicuna_v1.1" \
    --main_task_templ "CEE_vanilla" \
    --bf16 True \
    --remove_unused_columns False \
    --output_dir output/answer_42 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --metric_for_best_model "dev_macro_f1" \
    --greater_is_better "True" \
    --save_total_limit 10 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --num_beams 1 --temperature 1e-5 --top_k 50 --top_p 1.0 --max_new_tokens 1024 \
    --seed 42 \
    --save_predictions True --save_metrics True

# cd /data4/hzp/ECR-Chain/
# bash EmoReasonLlama/scripts/lora_train_answer.sh 0,1