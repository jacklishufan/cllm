MODEL_PATH=ckpt/consistency-llm-7b-math
TARGET_MODEL_PATH=ckpt/Abel-7B-001
CUDA_VISIBLE_DEVICES=0 bash eval/gsm8k/speedup.sh $MODEL_PATH $TARGET_MODEL_PATH 16 --data_size 10 ${@}