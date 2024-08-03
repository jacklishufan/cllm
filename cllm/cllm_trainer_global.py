import transformers
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import wandb
import random
from torch.utils.data import DataLoader
from tqdm.cli import tqdm
from utils import set_casual
# from utils import set_casual
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

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
            if len( inputs['labels_ids'].shape) == 3:
                inputs['labels_ids'] =  inputs['labels_ids'].squeeze(1) # hack
                
            labels = inputs['labels_ids']
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
        # breakpoint()
        
        label_student_model_output = model(labels_in, attention_mask)

        attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(model.device)
        attention_mask = jacobian_trajectory[-1] != self.tokenizer.pad_token_id
        if not causal:
            set_casual(model.model,True)
        logits_last =  self.get_logits(model, jacobian_trajectory[-1].clone().detach(), attention_mask,
                                       )

        label_smoother = LabelSmoother(epsilon=0.1, ignore_index= -100)
        loss_ar = label_smoother(label_student_model_output, labels, shift_labels=True)
        loss_ar*=self.args.ar_loss_weight
        if self.args.qlora:
            assert loss_ar.requires_grad
            #loss_ar.requires_grad = True
        print(f'loss ar: {loss_ar} computed! performing backward pass...')
        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss_ar)

        ### compute Consistency loss (global) ###
        # random select one point from trajectory
        #i = random.choice(range(len(jacobian_trajectory))[:-1])
        
        i = torch.randint(low=0,high=len(jacobian_trajectory)-1,size=(bsz,))
        # breakpoint()
        input_ids = torch.stack([jacobian_trajectory[j][k] for k,j in enumerate(i)])
        timesteps = i / max_new_tokens * 1000
        attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(jacobian_trajectory[0].device)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        if not causal:
            set_casual(model.model,False)
        logits_i = self.get_logits(model, input_ids.clone().detach(), attention_mask,
                                    block_size=max_new_tokens,
                                    timesteps=timesteps,
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
        loss_global *= self.args.consistency_loss_weight
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