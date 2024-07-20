MODEL_DIR=/home/jacklishufan/Consistency_LLM/ckpt/deepseekcoder-7b-instruct-spider
OUTPUT_DIR='ar_generated_spider_pretrained.jsonl'
CUDA_VISIBLE_DEVICES=0 python eval/spider/inference.py \
    --model_dir $MODEL_DIR \
    --temperature 0.0 \
    --top_p 1.0 \
    --output_file_name  $OUTPUT_DIR\
    --dev_set "gsm8k" --prompt_type math-single --max_new_tokens_for_consistency 16 --max_tokens 1024 --use_consistency_decoding --lora "" ${@}