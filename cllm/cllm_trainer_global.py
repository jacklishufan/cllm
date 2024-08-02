import transformers
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import wandb
import random
from torch.utils.data import DataLoader
from tqdm.cli import tqdm
from utils import set_casual
import numpy as np
import torch.nn.functional as F
# from utils import set_casual
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def get_non_padding_indices(attention_mask,pad_idx):
    batch_size, seq_length = attention_mask.shape
    min_indices = []
    max_indices = []
    
    for i in range(batch_size):
        true_indices = torch.nonzero(~(attention_mask[i] == pad_idx), as_tuple=True)[0]
        if len(true_indices) == 0:
            min_indices.append(None)  # or some other placeholder indicating no valid tokens
            max_indices.append(None)  # or some other placeholder indicating no valid tokens
        else:
            min_indices.append(true_indices.min().item())
            max_indices.append(true_indices.max().item())
    
    return min_indices, max_indices


def mask_last_block(input_ids,start_ids, mask_ratio, block_size,vocab_size,pad_token_id,mask_token_id=None):
    batch_size, seq_length = input_ids.shape
    is_mask = torch.zeros_like(input_ids).bool()
    is_last_block =  torch.zeros_like(input_ids)
    mask = input_ids.clone()
    attention_mask = torch.ones_like(input_ids).bool()
    for i in range(batch_size):
        start = start_ids[i].item() # Find the last block of B tokens where attention_mask is True
        non_zeros = torch.nonzero(input_ids[i]==pad_token_id)
        if non_zeros.numel() == 0:
            end = seq_length
        else:
            end = non_zeros.min().item()
        n_blocks = np.ceil((end-start)/block_size)
        if n_blocks == 0:
            # sth must be wrong????
            continue
        block_idx = np.random.randint(0,n_blocks) # 0 to n_block - 1
        last_block_start = start + block_idx * block_size
        last_block_end = last_block_start + block_size
        last_block_end = min(last_block_end,seq_length)
        assert last_block_end >last_block_start 
        last_block_indices = list(range(last_block_start,last_block_end))
        num_tokens_to_mask = int(mask_ratio[i] * (last_block_end-last_block_start)) # Calculate the number of tokens to mask
        mask_indices = np.random.choice(last_block_indices, num_tokens_to_mask, replace=False)  # Randomly select tokens to mask within the last block
        if mask_token_id is None:
            random_ids = torch.randint(0, vocab_size, (num_tokens_to_mask,), dtype=torch.long)  # Replace selected tokens with random IDs between 0 and VOCAB_SIZE
            mask[i, mask_indices] = random_ids.to(mask)
        else:
            mask[i, mask_indices] = mask_token_id
        mask[i,last_block_end:]=pad_token_id
        is_mask[i,mask_indices] = True
        is_last_block[i,last_block_start:last_block_end] = True
        attention_mask[i,last_block_end:]=False
    return mask,is_mask,attention_mask,is_last_block

class CllmTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.train_step_cnt = 0
        self.max_new_tokens = args.max_new_tokens
        self.use_gt_labels = args.use_gt_labels

    def training_step(self, model, inputs):
        self.train_step_cnt += 1
        self.model.train()
        return self.consistency_training_step(model, inputs)

    def consistency_training_step(self, model, inputs):

        max_new_tokens = self.max_new_tokens      
        causal = self.model.config.causal
        jacobian_trajectory = inputs["jacobian_trajectory"]
        input_masks = inputs["attention_mask"]
        bsz = jacobian_trajectory[0].shape[0]
        eos_reached = torch.tensor([False] * bsz).to(model.device)

        ### tokens generated after <eos> are set to <pad>
        for i in range(len(jacobian_trajectory)):
            for j in range(bsz):
                trajectory_len = torch.sum(input_masks, dim=-1)
                # find the first accurate <EOS>
                eos_positions = torch.where(jacobian_trajectory[i][j, :(trajectory_len[j]-max_new_tokens)]==self.tokenizer.eos_token_id)[0]
                if len(eos_positions)==0:
                    continue
                # otherwise, set tokens coming after the accurate <EOS> as pad 
                eos_reached[j] = True
                trajectory_copy = jacobian_trajectory[i].clone().detach()
                eos_pos = eos_positions[0]
                trajectory_copy[j, int(eos_pos)+1:] = self.tokenizer.pad_token_id
                jacobian_trajectory[i] = trajectory_copy  

        ### compute AutoRegression loss ###
        # use labels to avoid pattern collapse
        if self.use_gt_labels:
            labels = inputs['labels_ids']
        else:
            if self.args.use_full_seq:
                labels = inputs['complete_teacher_output_ids']
            else:
                labels = inputs['teacher_output_ids']
        # TODO: check if it's right when batch size > 1
        if isinstance(labels,torch.Tensor):
            labels = labels.to(model.device)
        else:
            labels = torch.tensor(labels).to(model.device)
        attention_mask = torch.full_like(labels, 1).to(model.device)

        labels_in = labels.clone()
        labels_in[labels==-100]=self.tokenizer.pad_token_id
        if self.args.ar_loss_weight > 0:
            label_student_model_output = model(labels_in, attention_mask)

            attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(model.device)
            attention_mask = jacobian_trajectory[-1] != self.tokenizer.pad_token_id
            if not causal:
                set_casual(model.model,True)

            label_smoother = LabelSmoother(epsilon=0.1, ignore_index= -100)
            
            loss_ar = label_smoother(label_student_model_output, labels, shift_labels=True)
            loss_ar = loss_ar * self.args.ar_loss_weight
            if self.args.qlora:
                assert loss_ar.requires_grad
                #loss_ar.requires_grad = True
            print(f'loss ar: {loss_ar} computed! performing backward pass...')
            with self.accelerator.accumulate(model):
                self.accelerator.backward(loss_ar)
        else:
            loss_ar = torch.tensor(0.0).to(model.device)

        ### compute Consistency loss (global) ###
        # random select one point from trajectory
        #i = random.choice(range(len(jacobian_trajectory))[:-1])
        
        
        i = torch.randint(low=0,high=len(jacobian_trajectory)-1,size=(bsz,))
        # breakpoint()
        if self.args.use_mask_diffusion:
            
            i = torch.randint(low=0,high=1000,size=(bsz,))
            #input_ids = inputs['labels_ids'].clone()
            # input_ids = inputs['labels_ids'].clone()
            # input_ids[input_ids==-100]=self.tokenizer.pad_token_id
            
            raw_input_ids = jacobian_trajectory[-1]
            input_ids = raw_input_ids.clone()
            # do masking
            start_pos = inputs['prompt_ids_len']
            # block_pos = start_pos + inputs['jacobian_itr_id'] * max_new_tokens
            # end_pos = block_pos + max_new_tokens
            if self.args.mask_policy == 'uniform':
                mask_input_ids,is_mask,attention_mask,is_last_block = mask_last_block(
                    input_ids,start_pos,i/1000,max_new_tokens,self.tokenizer.vocab_size,self.tokenizer.pad_token_id
                )
                loss_mask = attention_mask[:,1:] 
            elif self.args.mask_policy == 'mask':
                assert self.tokenizer.unk_token_id is not None
                mask_input_ids,is_mask,attention_mask,is_last_block = mask_last_block(
                    input_ids,start_pos,i/1000,max_new_tokens,self.tokenizer.vocab_size,self.tokenizer.pad_token_id,self.tokenizer.unk_token_id
                )
                loss_mask = (is_mask * attention_mask)[:,1:] 
            else:
                raise NotImplementedError(f"Policy {self.args.mask_policy} is invalid")
            timesteps = i #
            logits_i = self.get_logits(model, mask_input_ids.clone().detach(), attention_mask,
                                    block_size=max_new_tokens,
                                    timesteps=timesteps,
                                    is_last_block=is_last_block
            )
            #labels = input_ids[1:]
            # loss_global = F.cross_entropy(
            #     logits_i[:,:-1].permute(0,2,1),input_ids[:,1:],reduction='none',label_smoothing=0.1
            # )
            logits_last =  self.get_logits(model, raw_input_ids.clone().detach(), attention_mask,
                                       )
            for j in range(bsz):
                end_of_mask_position = torch.where(input_ids[j, 1:] != raw_input_ids[j, 1:])[0]
                if len(end_of_mask_position)==0:
                    output_mask[j, :] = True
                else:
                    output_mask[j, :end_of_mask_position[0]] = True
            
            loss_global = self.soft_cross_entropy(
                        logits_i[..., :-1, :].float(), # logits generated by the last token is dropped
                        logits_last[..., :-1, :].to(logits_i.device).clone().detach().float(),
                        output_mask.to(logits_i.device)
            )
                        
            # print(loss_mask.sum(-1,keepdim=True))
            # breakpoint()
            loss_global = (loss_global * loss_mask.to(loss_global)).mean(-1) /( loss_mask.sum(-1) +1) * loss_global.shape[-1]
            #TODO: add time weight?
            loss_global = loss_global.mean()
            loss_global = loss_global * self.args.consistency_loss_weight
        else:
            start_pos = inputs['prompt_ids_len']
            if type(start_pos) == list:
                start_pos = start_pos[0]
            block_pos = start_pos + inputs['jacobian_itr_id'] * max_new_tokens
            end_pos = block_pos + max_new_tokens
            input_ids = torch.stack([jacobian_trajectory[j][k] for k,j in enumerate(i)])
            is_last_block = torch.zeros_like(input_ids)
            logits_last =  self.get_logits(model, jacobian_trajectory[-1].clone().detach(), attention_mask,
                                       )
            timesteps = i / max_new_tokens * 1000
            attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(jacobian_trajectory[0].device)
            attention_mask = input_ids != self.tokenizer.pad_token_id
            if not causal:
                set_casual(model.model,False)
            logits_i = self.get_logits(model, input_ids.clone().detach(), attention_mask,
                                        block_size=max_new_tokens,
                                        timesteps=timesteps,
                                        is_last_block=is_last_block
            )

            output_mask = input_ids[..., 1:] == self.tokenizer.pad_token_id
            # We do not calculate the cross entrophy of same logits to alleviate misleading gradients
            for j in range(bsz):
                end_of_mask_position = torch.where(input_ids[j, 1:] != jacobian_trajectory[-1][j, 1:])[0]
                if len(end_of_mask_position)==0:
                    output_mask[j, :] = True
                else:
                    output_mask[j, :end_of_mask_position[0]] = True
            
            loss_global = self.soft_cross_entropy(
                        logits_i[..., :-1, :].float(), # logits generated by the last token is dropped
                        logits_last[..., :-1, :].to(logits_i.device).clone().detach().float(),
                        output_mask.to(logits_i.device)
            )
            loss_global = loss_global * self.args.consistency_loss_weight
        if self.args.qlora:
            assert loss_global.requires_grad
            #loss_global.requires_grad = True
        print(f'loss global {loss_global} computed! performing backward pass...')
        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss_global)
        
        if self.args.local_rank == 0:
            wandb.log({"ar loss": loss_ar})
            wandb.log({"consistency loss": loss_global})

        # sync processes
        torch.distributed.barrier()
        # total loss = ar_loss + consistency_global_loss
        loss = loss_ar.detach() + loss_global.detach()

        return loss
    

    def log(self, logs):
        # Remove the 'loss' entry with value 0 before calling the superclass method
        if 'loss' in logs and logs['loss'] == -1:
            del logs['loss']

        # Call the original `log` method of the `Trainer` class
        super().log(logs)

    def get_train_dataloader(self):
        # Create custom DataLoader with shuffle set to False
        shuffle = True
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "shuffle": shuffle,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "collate_fn":self.data_collator
        }

        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

    ###################### Helper Functions #############################
    def soft_cross_entropy(self, predicts, targets, padding_mask):
        # TODO: support batch_size >1 here.
        if (~padding_mask).sum() == 0:
            return 0*predicts[0][0][0]
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy

    def get_logits(self, model, input_ids, attention_mask,**kwargs):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        ).logits


    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        eval_loader = self.get_eval_dataloader()
        self.model.eval()
        total_eval_loss = 0.0
        total_eval_loss_batch = 0
        _idx = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                _idx +=1
                if _idx > 100:
                    break
                labels_ids = batch['labels_ids']
                input_ids = labels_ids.clone()
                attention_mask = ~(input_ids == -100)
                attention_mask = attention_mask.long()
                for idx,prompt_len in enumerate(batch['sources_len'].tolist()):
                    labels_ids[idx,:prompt_len] = -100
                input_ids[input_ids == -100] = self.tokenizer.pad_token_id
                logits = self.get_logits(self.model,input_ids,attention_mask)
                label_smoother = LabelSmoother(epsilon=0.0, ignore_index= -100)
                loss_ar = label_smoother(dict(logits=logits), labels_ids, shift_labels=True)
                total_eval_loss += loss_ar.item()
                total_eval_loss_batch += 1
            total_eval_loss = torch.tensor(total_eval_loss).to(self.accelerator.device)
            total_eval_loss_batch = torch.tensor(total_eval_loss_batch).to(self.accelerator.device)
            total_eval_loss_all_rank = self.accelerator.reduce(total_eval_loss)
            total_eval_loss_batch = self.accelerator.reduce(total_eval_loss_batch)
            nll = total_eval_loss_all_rank / (total_eval_loss_batch+1e-9)
            ppl = torch.exp(nll)
            payload = dict(
                nll=nll.item(),
                ppl=ppl.item()
            )
            if wandb.run is not None:
                wandb.log({metric_key_prefix+k:v for k,v in payload.items()})
    torch.cuda.empty_cache()