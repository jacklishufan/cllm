from transformers import LlamaModel,LlamaConfig
import torch
from tqdm.cli import tqdm
from torch import nn
import numpy as np
from itertools import product
import wandb
import pandas as pd
import argparse
import time
from transformers.cache_utils import Cache, DynamicCache
# config = LlamaConfig.from_json_file('/home/jacklishufan/Consistency_LLM/config/llama0.json')
# config.hidden_size=HIDDEN_SIZE
# config.num_hidden_layers = NUM_LAYERS
# config.intermediate_size = INTERMEDIATE_ZIE
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size',default=4096,type=int)
parser.add_argument('--num_layers',default=32,type=int)
parser.add_argument('--mlp_dim',default=12288,type=int)
parser.add_argument('--prompt_len',default=256,type=int)


args= parser.parse_args()
HIDDEN_SIZE=args.hidden_size
NUM_LAYERS=args.num_layers
INTERMEDIATE_ZIE=args.mlp_dim
# f'Exp-HiddenSize{HIDDEN_SIZE}-{NUM_LAYERS}Layers-MLP_DIM{INTERMEDIATE_ZIE}'
print("Init0:")

wandb.init(mode="disabled")
print("Init1:")

DTYPE=torch.bfloat16 
device = 'cuda'
print("Init:")
model = LlamaModel.from_pretrained('/home/bootstrap/jacklishufan/Consistency_LLM/ckpt/deepseekcoder_7b_codesearch_net_python',torch_dtype=DTYPE)
#model = #LlamaModel._from_config(config,torch_dtype=DTYPE,attn_implementation="flash_attention_2",)#.to(device)
VOCAB_SIZE=model.config.vocab_size
def set_casual(model,value):
    for layer in model.layers:
        layer.self_attn.is_causal = value
set_casual(model,False)
wandb.run.summary['model_size']=model.num_parameters()
# wandb.run.summary.update()
print(f"Done: {model.num_parameters()}")


class AvgMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0
        
    def log(self,x,c=1):
        self.sum += x
        self.count +=c



model.to(device)


lm_head = nn.Linear(model.config.hidden_size,VOCAB_SIZE).to(device).to(DTYPE)


results = []
step_choices = [product]
MAX_STEP = 100
# block_sizes = reversed([1,16,32,54,128,200,256,512,700,800,1000])
# seq_len = list(reversed([128,256,512,1024,2048]))
block_sizes = [256]
seq_len = [1024]
TARGET_LENGTH = 529
BLOCK_SIZE=256
L_PROMPT= args.prompt_len
N_EXPS = 1

# def delete_false_key_value(
#         self,
#         num_of_false_tokens,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
   
#         for layer_idx in range(len(self.key_cache)):
#             self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
#             self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]


for TARGET_LENGTH,BLOCK_SIZE in product(seq_len,block_sizes):
    print(TARGET_LENGTH,BLOCK_SIZE)
    N_BLOCKS = np.ceil(TARGET_LENGTH / BLOCK_SIZE).astype(int)
    max_step = min(MAX_STEP,BLOCK_SIZE)
    if max_step == 1:
        steps = [1]
    else:
        steps = list(range(1,max_step,max_step//10))
        if max_step not in steps:
            steps.append(max_step)
    steps = [51]
    for N_STEPS in reversed(steps):
        print( dict(
            target_length=TARGET_LENGTH,
            block_size=BLOCK_SIZE,
            n_step=N_STEPS,
            n_blocks=N_BLOCKS,
            ))
        total_time = 0
        total_tokens = 0
        avg_latency = AvgMeter()
        for _ in range(N_EXPS):
            torch.cuda.empty_cache()
            x = (torch.rand(L_PROMPT)*VOCAB_SIZE).int()[None].to(device)
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            first_output = torch.cuda.Event(enable_timing=True)
            start.record()
            cache = None
            n_forward = 0
            with torch.no_grad():
                for i in tqdm(range(N_BLOCKS),position=2):
                    rand_init =  (torch.rand(BLOCK_SIZE)*1024).int()[None].to(device)
                    x = torch.cat([x,rand_init],dim=-1)
                    for _ in tqdm(range(N_STEPS),position=1):
                        n_forward += 1
                        start0 = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize() ## simulate cllm check
                        t00 = time.time()
                        end0 = torch.cuda.Event(enable_timing=True)
                        start0.record()
                        if cache is not None:
                            print(cache[0][0].shape)
                        y = model(input_ids=x[:,-BLOCK_SIZE:] if cache is not None else x,past_key_values=cache,use_cache=True)
                        end0.record()
                        torch.cuda.synchronize() ## simulate cllm check
                        print(start0.elapsed_time(end0) * 1e-3)
                        if cache is None:
                            cache = y['past_key_values'] # prompt
                        x_new = y['last_hidden_state']
                        x_new = lm_head(x_new[:,-BLOCK_SIZE:])
                        x_new = x_new.argmax(-1)
                        x[:,-BLOCK_SIZE:]=x_new
                        torch.cuda.synchronize() ## simulate cllm check
                        t03 = time.time()
                        print(t03-t00)
                    if i == 0:
                            first_output.record()
                    cache = y['past_key_values']
                    print(cache[0][0].shape[2])      
            end.record()
            torch.cuda.synchronize()
            dt_decode = start.elapsed_time(end) * 1e-3
            dt_first = start.elapsed_time(first_output) * 1e-3
            avg_latency.log(dt_first)
            total_time += dt_decode
            if BLOCK_SIZE == 1:
                total_tokens += N_BLOCKS
            else:
                total_tokens +=( BLOCK_SIZE * N_BLOCKS - BLOCK_SIZE // 2 ) 
        spped = total_tokens / (total_time+1e-9)
        avg_latency = avg_latency.sum / (avg_latency.count + 1e-9)
        print(f"{N_STEPS}:: Speed is {spped} tk /s \t Avg Latency is {avg_latency}")
        payload =             dict(
            target_length=TARGET_LENGTH,
            block_size=BLOCK_SIZE,
            n_step=N_STEPS,
            n_blocks=N_BLOCKS,
            n_forward=n_forward,
            n_layers=NUM_LAYERS,
            hidden_size=HIDDEN_SIZE,
            mlp_dim=INTERMEDIATE_ZIE,
            spped=spped,
            total_tokens=total_tokens,
            total_time=total_time,
            latency=avg_latency
            )
        results.append(payload)
        wandb.log(
            dict(
                table=wandb.Table(dataframe=pd.DataFrame(results))
            )
        )
wandb.log(
    dict(
        table=wandb.Table(dataframe=pd.DataFrame(results))
    )
)
# print(prof)