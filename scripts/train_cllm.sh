# export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT=consistency_llm

model_path=$1
trajectory_file=$2
output_path=$3
n_token_seq_size=$4
qlora=False
# 2e-5
LR=2e-5
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=101 --rdzv_endpoint='localhost:5666' \
    --master_port 10000 \
    cllm/train_cllm_global.py \
    --target_model_path ${model_path} \
    --data_path ${trajectory_file} \
    --output_dir ${output_path} \
    --max_new_tokens ${n_token_seq_size} \
    --bf16 True \
    --report_to wandb \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 4 \
    --learning_rate $LR \
    --max_grad_norm 1.0 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --qlora ${qlora} ${@:5}
