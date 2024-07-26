MODEL_DIR=output/python5/checkpoint-6000
OUTPUT_DIR='eval/human-eval/output_cllm.jsonl'
CUDA_VISIBLE_DEVICES=4 python eval/human-eval/inference-hack.py \
    --model_dir $MODEL_DIR \
    --temperature 0.0 \
    --top_p 1.0 \
    --output_file_name  $OUTPUT_DIR\
    --dev_set "gsm8k" --prompt_type math-single --max_new_tokens_for_consistency 64 --max_tokens 1024 --use_consistency_decoding --lora "" ${@}

bash eval.sh