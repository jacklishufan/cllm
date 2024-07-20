OUTPUT_DIR=output/spider-32
MODEL_PATH=/home/jacklishufan/Consistency_LLM/ckpt/deepseekcoder-7b-instruct-spider
DATA=data/collected_jacobi_trajectory/cleaned_spider_jacobi_max_new_tokens32_augTrue_labels_True_max_seq_len_512.json
bash scripts/train_cllm.sh $MODEL_PATH $DATA  $OUTPUT_DIR 32