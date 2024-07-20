MODEL_DIR=output/spider-32
OUTPUT_DIR='cllm_generated_spider_32.jsonl'
CUDA_VISIBLE_DEVICES=1 python eval/spider/inference.py \
    --model_dir $MODEL_DIR \
    --temperature 0.0 \
    --top_p 1.0 \
    --output_file_name  $OUTPUT_DIR\
    --dev_set "gsm8k" --prompt_type math-single --max_new_tokens_for_consistency 32 --max_tokens 1024 --use_consistency_decoding --lora "" ${@}