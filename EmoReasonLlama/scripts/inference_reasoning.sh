export PYTHONPATH=/data4/hzp/ECR-Chain

gpu_id=$1

CUDA_VISIBLE_DEVICES=${gpu_id} python3 EmoReasonLlama/run/inference_by_generate.py \
    --dataset 'CEEBackwardFilterCausalLMDataset' --set_name 'test' \
    --raw_data_path '/data4/hzp/ECR-Chain/RECCON_all_data.json' \
    --conv_templ_type 'vicuna_v1.1' --main_task_templ 'CEE_simplify_1shot' \
    --example_id 'tr_264_6' \
    --max_remain_utts 20 \
    --padding_side 'right' \
    --num_beams 1 --temperature 1e-5 --top_k 50 --top_p 1.0 --max_new_tokens 1024 \
    --model-path '/data4/hzp/pretrained_models/vicuna/vicuna-7b-v1.3/' \
    --lora_path '[checkpoint_dir_of_your_trained_model]' \
    --device cuda \
    --num-gpus 1 \
    --save_predictions --save_metrics

# cd /data4/hzp/ECR-Chain/
# bash EmoReasonLlama/scripts/inference_reasoning.sh 1