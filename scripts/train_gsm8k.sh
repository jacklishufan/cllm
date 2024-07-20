OUTPUT_DIR=output/math-nolora
MODEL_PATH=/home/jacklishufan/Consistency_LLM/ckpt/Abel-7B-001
MODEL_PATH=output/math-nolora
MODEL_PATH=output/math-nolora-ep2
DATA=/home/jacklishufan/Consistency_LLM/data/collected_jacobi_trajectory/cleaned_gsm8k_train_jacobi_max_new_tokens16_augTrue_labels_True_max_seq_len_1024_rank_all.json
bash scripts/train_cllm.sh $MODEL_PATH $DATA  $OUTPUT_DIR 16 --num_train_epochs 1 ${@}