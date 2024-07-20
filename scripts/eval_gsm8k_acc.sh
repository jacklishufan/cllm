MODEL_DIR=ckpt/consistency-llm-7b-math
MODEL_DIR=/home/jacklishufan/Consistency_LLM/ckpt/Abel-7B-001
ADAPTOR_DIR=/home/jacklishufan/Consistency_LLM/output/math
MODEL_DIR=/home/jacklishufan/Consistency_LLM/output/math-nolora
CUDA_VISIBLE_DEVICES=1 python eval/gsm8k/acc.py --model_dir $MODEL_DIR --temperature 0.0 --top_p 1.0 --output_file_name 'cllm_generated_gsm8k_nolora.jsonl' \
--dev_set "gsm8k" --prompt_type math-single --max_new_tokens_for_consistency 16 --max_tokens 1024 --use_consistency_decoding --lora "" ${@}