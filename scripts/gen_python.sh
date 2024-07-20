filename=data/raw_data/gsm8k_train.jsonl
model_path=/home/jacklishufan/Consistency_LLM/ckpt/deepseekcoder_7b_codesearch_net_python
max_new_tokens=$1
max_new_seq_len=$2

accelerate launch --num_processes=2 data/generate_trajectory_distributed.py \
	--filename ${filename} \
	--model ${model_path} \
	--max_new_tokens ${max_new_tokens} \
	--max_new_seq_len ${max_new_seq_len} \
    --data_size 999999 \
