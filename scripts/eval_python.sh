MODEL_PATH=ckpt/consistency-llm-7b-codesearchnet
TARGET_MODEL_PATH=ckpt/deepseekcoder_7b_codesearch_net_python
CUDA_VISIBLE_DEVICES=0 bash eval/code-search-net/speedup.sh $MODEL_PATH $TARGET_MODEL_PATH 32 --data_size 10