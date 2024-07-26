MODEL_DIR=output/python4/checkpoint-2500
#MODEL_DIR=ckpt/deepseekcoder_7b_codesearch_net_python
OUTPUT_DIR='eval/human-eval/output_cllm.jsonl'
CUDA_VISIBLE_DEVICES=4 python eval/human-eval/inference.py \
    --model_dir $MODEL_DIR \
    --temperature 0.0 \
    --top_p 1.0 \
    --output_file_name  $OUTPUT_DIR \
    --dev_set "gsm8k" --prompt_type math-single --max_new_tokens_for_consistency 1 --max_tokens 1024 --lora "" ${@}