import transformers
import os
import re
import json
import jsonlines
import argparse
import torch
from tqdm import tqdm
import sys
import pdb
import time
import random
import numpy as np
from human_eval.data import write_jsonl, read_problems
# from math_normalization import *
import ast
from dataclasses import dataclass, field
import json
import math
import pathlib
import functools
from typing import Dict, Optional, Sequence, List, Tuple
import random
from tqdm import tqdm
import torch.nn.functional as F
import sqlite3
import time
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother, get_module_class_from_name
from fastchat.model.model_adapter import get_conversation_template
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
import sys
sys.path.append('/home/bootstrap/jacklishufan/Consistency_LLM/')
from cllm.modeling import LLamaForMaskedDiffusion

def delete_false_key_value(
        self,
        num_of_false_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
   
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]


import torch.nn.functional as F
from transformers import LlamaModel,LlamaForCausalLM
import argparse

def get_usable_length(cache):
    if cache is None:
        return 0
    else:
        return len(cache[0][0])
@torch.inference_mode()
def jacobi_forward_profiling(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
):
    
    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        # self.model is the instance of class LlamaModel
        inputs_embeds = self.model.embed_tokens(input_ids)
        past_key_values_length = 0
        if use_cache:
            # if past_key_values is None:
            #     past_key_values_length =
            use_legacy_cache = not isinstance(past_key_values, Cache)
            
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length) 

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa :
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        for decoder_layer in self.model.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        hidden_states = self.model.norm(hidden_states)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
        first_correct_token = predict_next_tokens[:, -1]
        return next_decoder_cache.to_legacy_cache(), first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0
        next_point = input_ids
        jacobian_trajectory.append(next_point)

        iter_counter = 0
        base_cache = past_key_values
        timestep = torch.tensor([1]).long().cuda()
        while True:
            past_key_values = base_cache
            current_point = next_point
            torch.cuda.synchronize()
            t00 = time.time()
            y = self(input_ids=current_point,past_key_values=past_key_values,use_cache=True,block_size=max_new_tokens,timesteps=timestep)
            # breakpoint()
            timestep += 1
            output_ids = y.logits.argmax(-1) # self.lm_head(y.last_hidden_state).argmax(-1)
            next_point= torch.cat((current_point[0, 0].view(1,-1), output_ids[0, :seq_length-1].view(1,-1)), dim=-1)
            torch.cuda.synchronize()
            t1 = time.time()
            torch.cuda.synchronize()
            t2 = time.time()
            if torch.all(torch.eq(current_point, next_point)).item() or iter_counter >max_new_tokens:    
            #if iter_counter == 50:
                #print('Successfully break!')
                #print(next_point)
                first_correct_token = output_ids[:,-1]
                break
            #breakpoint()
            past_key_values = base_cache
            #print(base_cache[0][0].shape)
            #delete_false_key_value(past_key_values,seq_length)
            torch.cuda.synchronize()
            t3 = time.time()
            #print(t3-t00)
            #print(t1-t0,t2-t1,t3-t2,t0-t00)
            iter_counter += 1

        return jacobian_trajectory[:-1], next_point, first_correct_token, iter_counter,y.past_key_values
    
def extract_docstring(function_code):
    """
    Extracts the docstring from a given Python function code.

    Parameters:
    function_code (str): A string containing the code of the Python function.

    Returns:
    str: The extracted docstring, or None if no docstring is found.
    """
    # Parse the function code into an AST
    tree = ast.parse(function_code)
    
    # Find the function definition node
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return ast.get_docstring(node)
    
    return None

def remove_unit_tests_from_docstring(docstring):
    """
    Removes unit tests from a docstring that are in the '>>>' format.

    Parameters:
    docstring (str): The original docstring containing unit tests.

    Returns:
    str: The docstring with unit tests removed.
    """
    # Define the regex pattern to match '>>>' lines and the following answer lines
    pattern = r'>>>.*\n(?:.*\n)*?(?=(>>>|$))'
    
    # Remove all matches of the pattern
    cleaned_docstring = re.sub(pattern, '', docstring)
    
    return cleaned_docstring.strip()

class AvgMeter:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0
        self.flag = 0
        self.tokens = 0
        
    def add(self,x):
        self.sum+=x
        self.count+=1
    def get_value(self):
        return self.sum/ ( self.count+1e-9)
    
def load_json(fp):
    with open(fp) as f:
        data = json.loads(f.read())
    return data

def jacobi_generate(inputs, model, tokenizer, max_new_tokens, max_new_seq_len,decode_step):
    converge_step = []
    forward_times = 0

    all_jacobian_trajectory = []
    prompt_len = torch.sum(inputs['attention_mask'], dim=-1)
    generation = inputs['input_ids']
    ### prefill the kv-cache
    past_key_values, first_correct_token = jacobi_forward_profiling(model,input_ids=inputs['input_ids'], max_new_tokens=max_new_tokens, past_key_values=None, use_cache = True, prefill_phase = True)
    ### generation phase
    itr = 0
    eos_reached = False
    while True:
        itr+=1
        bsz = 1 # only support batch_size = 1 now
        # randomly initialize the first point of jacobian trajectory
        random_point = torch.tensor(random.choices(generation[0], k=(max_new_tokens-1)), device="cuda").view(1,-1)
        input_ids = torch.cat((first_correct_token.view(1,-1), random_point),dim=-1)
        jacobian_trajectory, n_gram_generation, first_correct_token, iter_steps,past_key_values = jacobi_forward_profiling(model,input_ids=input_ids, max_new_tokens=max_new_tokens, past_key_values=past_key_values, use_cache = True, prefill_phase = False)
        decode_step.add(iter_steps)
        forward_times += iter_steps
        #all_jacobian_trajectory.append(jacobian_trajectory)
        eos_positions = torch.where(n_gram_generation[0]==tokenizer.eos_token_id)[0]

        if len(eos_positions)>0:
            eos_reached = True
        
        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_id 
        generation = torch.cat((generation, n_gram_generation), dim=-1)
        if eos_reached or itr*max_new_tokens > max_new_seq_len:
            break
    
    # to support bsz > 1
    # print(generation.shape)
    converge_step.append(forward_times / itr)
    return generation[0, prompt_len:][None]#, converge_step, all_jacobian_trajectory


def consistency_generate(
    model,
    tokenizer,
    inputs,
    max_new_tokens,
    max_new_seq_len,
    decode_step,
    ):
    itr = 0
    cache = None
    # jacobi_forward_profiling(
    #     input_ids = inputs['input_ids']
    #     # input_masks = inputs['attention_mask'],
    #     max_new_tokens=max_new_tokens,
    #     past_key_values=None,
    #     use_cache = True, 
    #     prefill_phase = True
    #     use_cache=True
    # )
    while True:
        if itr == 0:
            input_ids = inputs['input_ids']
            input_masks = inputs['attention_mask']
        else:
            input_masks = torch.ones_like(input_ids).to(input_ids.device)
            for j in range(bsz):
                input_masks[j][torch.sum(inputs["attention_mask"], dim=-1)[j] + itr*max_new_tokens:] = 0

        bsz = input_ids.shape[0]
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        #generation,cache = get_jacobian_trajectory(model, tokenizer, input_ids, input_masks, max_new_tokens,decode_step,cache)
        ### tokens generated after <eos> are set to <pad>
        for j in range(bsz):
            prompt_len = torch.sum(input_masks, dim=-1)
            eos_positions = torch.where(generation[j]==tokenizer.eos_token_id)[0]
            if len(eos_positions)==0:
                # no EOS, continue to the next item in the batch
                total_token_len = prompt_len + max_new_tokens
                continue
            # otherwise, set tokens coming after EOS as pad 
            eos_reached[j] = True
            total_token_len = int(eos_positions[0])

        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_ids
        itr+=1      
        if all(eos_reached) or itr*max_new_tokens >= max_new_seq_len:
            return generation[0, :total_token_len]
        input_ids = generation

@torch.inference_mode()
def get_jacobian_trajectory(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    decode_step,
    cache = None
):

    bsz = input_ids.shape[0] 
    prompt_len = [torch.sum(t) for t in attention_mask]
    max_prompt_len = max(prompt_len)
    total_len = max_prompt_len + max_new_tokens

    # initialize the first point of jacobian trajectory
    tokens = torch.full((bsz, total_len), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
    for i in range(bsz):
        tokens[i, :] = torch.tensor(random.choices(input_ids[i][attention_mask[i]==1], k=total_len), dtype=torch.long, device="cuda")
        tokens[i, : prompt_len[i]] = torch.tensor(input_ids[i][: prompt_len[i]], dtype=torch.long, device="cuda")
    itr = 0
    next_generation = tokens
    generate_attention_mask = torch.full_like(next_generation, 1).to(tokens.device)
    while True:

        current_generation = next_generation
        with torch.no_grad():
            if cache is not None:
                assert cache[0][0].shape[2] ==  input_ids.shape[1]
                y = model(current_generation[:,-max_new_tokens:],use_cache=True,past_key_values=cache,return_dict=True)
                breakpoint()
            else:
                y = model(current_generation,use_cache=True,past_key_values=cache,return_dict=True)
                n_input = input_ids.shape[1]
                cache = []
                for cache_k,cache_v in y.past_key_values:
                    cache.append((cache_k[:,:,:n_input],cache_v[:,:,:n_input]))
                logits = y.logits
                next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)

        # hold prompt unchanged and update generated tokens
        for i in range(bsz):
            next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, -1-max_new_token:-1]), dim=0)
        if torch.all(torch.eq(next_generation, current_generation)).item():
            print(f"Iteration steps: {itr}")
            decode_step.add(itr)
            cache =  y.past_key_values
            return next_generation,cache # right generation is saved twice so we delete the last element of trajectory list
        itr+=1
        

def get_results(pred_file, dev_set):
    def test_answer(pred_str, ans_str):
        pattern = "#### (.*)$"

        if "Question" in pred_str:
            pred_str = pred_str.split("Question")[0]

        preds = re.findall(pattern, pred_str)
        pred = preds[-1] if len(preds) >= 1 else ""
        if "</s>" in pred:
            pred = pred[:-4]

        gold = ans_str
        pred = normalize_final_answer(pred)
        gold = normalize_final_answer(gold)
        return check_sympy_equivalence(gold, pred), pred, gold

    def parse_pred_ans(preds_str, golds_str, properties_list):
        num_q = 0
        acc = 0
        results = []
        preds = []
        golds = []
        correct_table = {}
        cnt_table = {}
        source_set = set()
        for pred_str, gold_str, properties in tqdm(zip(preds_str, golds_str, properties_list), total=len(preds_str)):
            num_q += 1
            result, pred, gold = test_answer(pred_str, gold_str)
            results.append(result)
            preds.append(pred)
            golds.append(gold)
            if result:
                acc += 1
            source = properties['source']
            tag = properties['tag']
            source_set.add(source)
            if source not in correct_table.keys():
                correct_table[source] = 1 if result else 0
                cnt_table[source] = 1
            else:
                correct_table[source] = (correct_table[source] + 1) if result else correct_table[source]
                cnt_table[source] += 1
            for key in tag.keys():
                value = tag[key]
                value = source+","+key+"__"+value
                if value not in correct_table.keys():
                    correct_table[value] = 1 if result else 0
                    cnt_table[value] = 1
                else:
                    correct_table[value] = (correct_table[value] + 1) if result else correct_table[value]
                    cnt_table[value] += 1
        print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
        acc_table = {}
        for key in correct_table.keys():
            acc_table[key] = correct_table[key] / cnt_table[key]
        acc_table = list(zip(acc_table.keys(), acc_table.values()))
        acc_table.sort(key=lambda x: x[0])
        for key, acc in acc_table:
            if key in source_set:
                print(key+" : "+str(acc))
            else:
                print("    " + key.split(",")[-1]+ " : " + str(acc))
        return results, preds, golds

    if dev_set in ['all', 'gsm8k', 'math', 'mathgpt', 'gsm8k_robust']:
        golds_str = []
        properties = []
        with open(f'eval/gsm8k/test.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if dev_set != "all":
                    if json.loads(line)['source'].lower() == dev_set:
                        golds_str.append(json.loads(line)['target'])
                        properties.append({"source": json.loads(line)['source'], "tag": json.loads(line)['tag']})
                else:
                    golds_str.append(json.loads(line)['target'])
                    properties.append({"source": json.loads(line)['source'], "tag": json.loads(line)['tag']})
        preds_str = []
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                preds_str.append(json.loads(line)['response'])
        results, preds, golds = parse_pred_ans(preds_str, golds_str, properties)
        with open(pred_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        for i, line in enumerate(data):
            line['pred'] = preds[i]
            line['gold'] = golds[i]
            line['result'] = results[i]

        # Save the updated list of dictionaries back to the jsonl file
        with open(pred_file, 'w') as file:
            for item in data:
                file.write(json.dumps(item) + '\n')

    else:
        raise NotImplementedError("Evaluation not supported.")


def get_raw_inputs(dev_set):
    # in this function, we will get the raw queries for a target dev set
    data = []
    if dev_set in ['all', 'gsm8k', 'math', 'mathgpt', 'gsm8k_robust']:
        with open(f'eval/gsm8k/test.jsonl') as f:
            for line in jsonlines.Reader(f):
                data.append(line)
        if dev_set != 'all':
            data = [line for line in data if line['source'].lower() == dev_set]
    else:
        raise ValueError

    prompt_list = [line['question'] for line in data]
    return prompt_list


prompt_mapping = {
    "math-single": "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
}
MAX_FILL = '''You are given a rectangular grid of wells. Each row represents a single well,
and each 1 in a row represents a single unit of water.
Each well has a corresponding bucket that can be used to extract water from it, 
and all buckets have the same capacity.
Your task is to use the buckets to empty the wells.
Output the number of times you need to lower the buckets.
'''
def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
    
if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--lora', type=str, default='')
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--output_file_name', type=str, default='output.json')
    parser.add_argument('--stop', type=str, nargs='+', default=[], help="you can pass one or multiple stop strings to halt the generation process.")
    parser.add_argument('--dev_set', type=str, default='all')
    parser.add_argument('--prompt_type', type=str, default='math-single')
    parser.add_argument('--sample_num', type=int, default=-1, )
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--max_num_batched_tokens', type=int, default=2048)
    parser.add_argument('--use_temporal_embedding',action='store_true')
    parser.add_argument('--causal',type=bool,default=True)
    parser.add_argument(
        "--use_consistency_decoding",
        action="store_true",
        help="Whether to use consistency decoding",
    )
    parser.add_argument(
        "--max_new_tokens_for_consistency",
        type=int,
        default=16,
        help="The n-gram for consistency decoding.",
    ) 
    args = parser.parse_args()
    max_new_token = args.max_tokens
    if args.eval_only == False:
        # part 1 we set the model and tokenizer
        #model_cls = LLamaForMaskedDiffusion
        config = transformers.AutoConfig.from_pretrained(args.model_dir)
        config.causal = args.causal
        config.use_temporal_embedding = args.use_temporal_embedding
        model = LLamaForMaskedDiffusion.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=True,
            device_map='cuda',
        )
        if args.lora:
            model.load_adapter(args.lora)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.lora or args.model_dir,
            padding_side="right",
            # use_fast=False,
        )
        print('>>>>>> model and tokenizer loaded')

        # part 2 we prepare raw queries and wrap them with target prompt
        # raw_queries = get_raw_inputs(args.dev_set)
        # prompt = prompt_mapping[args.prompt_type]
        # processed_prompts = [prompt.format(input=query) for query in raw_queries]
        # processed_prompts = processed_prompts[:args.sample_num] if args.sample_num > 0 else processed_prompts
        #raw_data = load_json('eval/spider/test_prompt_ids.json')
        raw_data = read_problems()
        outputs = []
        all_time = 0
        all_tokens = 0
        all_tokens_t = 0
        decode_step = AvgMeter()
        context_len = AvgMeter()
        i = 0
        # part 3 we generate answers
        for task_id,processed_prompt in tqdm(raw_data.items()):
            i += 1
            question =processed_prompt['prompt']

            instruction = extract_docstring(question)
            if not instruction:
                if 'def max_fill(grid, capacity):' in question:
                    instruction = MAX_FILL
            instruction = remove_unit_tests_from_docstring(instruction)

            assert instruction


            prompt = "Please generate code based on the following doc:\n" + \
            f'''### Instruction:
{instruction}
### Response:\n{question}'''

            prompt = _tokenize_fn([prompt],tokenizer)
            input_ids = prompt['input_ids'][0].long()[None]
            attention_mask = torch.ones_like(input_ids)
            payload = dict(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device)
            )
            t0 = time.time()
            context_len.add(len(input_ids[0]))
            if args.use_consistency_decoding:
                # output_ids = consistency_generate(
                #     model,
                #     tokenizer,
                #     payload,
                #     max_new_tokens=args.max_new_tokens_for_consistency,
                #     max_new_seq_len=max_new_token,
                #     decode_step=decode_step
                # )
                output_ids = jacobi_generate(
                    payload,model,tokenizer,max_new_tokens=args.max_new_tokens_for_consistency,
                    max_new_seq_len=max_new_token,
                    decode_step=decode_step
                )
                output_ids = output_ids#.unsqueeze(dim=0)
            else:
                output_ids = model.generate(
                    input_ids.cuda(),
                    do_sample=False,
                    # temperature=args.temperature,
                    max_new_tokens=max_new_token,
                )
                decode_step.add(len(output_ids[0])-len(input_ids[0]))
            t1 = time.time()
            output_ids = output_ids[0]
            # if model.config.is_encoder_decoder:
            #     output_ids = output_ids[0]
            # else:
            #     output_ids = output_ids[0][len(input_ids[0]) :]
            try:
                output_ids = output_ids[:torch.where(output_ids==tokenizer.eos_token_id)[0].min()]
            except:
                pass
            
            generated_tokens = len(output_ids)
            # print(generated_tokens,max_new_token)
            #generated_tokens += max(max_new_token,generated_tokens)
            
            all_tokens += generated_tokens
            block_size = args.max_new_tokens_for_consistency
            all_tokens_t +=  np.ceil(generated_tokens/block_size)*block_size
            all_time += (t1 - t0)
            print(f"Throughput: {all_tokens/(all_time+1e-9)}  /{all_tokens_t/(all_time+1e-9)} tokens / s"
                  f"Token Utilization: {all_tokens/all_tokens_t}",
                  f"Avg Tokens: {all_tokens /i }"
                  f"Avg Step: {decode_step.get_value()}",
                  f"Context Len: {context_len.get_value()}"
                  )
            output = tokenizer.decode(
                output_ids,
              
            )#.split('<|EOT|>')[0]
            #print(question+output)
            # breakpoint()
            
            outputs.append({'task_id': task_id, 'completion': output})
        print('>>>>>> generation done')
        write_jsonl(args.output_file_name, outputs)
        # part 5 we save the results, always be {'id':id,'response':response}
        # if dir of output file is not exist, it will be created automatically
        print('>>>>>> writing prediction done')

    # # part 6 evaluate, I guess this should be done in a separate script
    # get_results(args.output_file_name, args.dev_set)
    print('>>>>>> evaluation done')


# CUDA_VISIBLE_DEVICES=0 acc.py --model_dir path_to_cllm --temperature 0.0 --top_p 1.0 --output_file_name 'cllm_generated_gsm8k.jsonl' --dev_set "gsm8k" --prompt_type math-single --max_new_tokens_for_consistency 16 --max_tokens 1024 --use_consistency_decoding