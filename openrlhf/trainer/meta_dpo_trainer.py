import os
from abc import ABC

import torch
from flash_attn.utils.distributed import all_gather
from torch.nn import functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import DPOLoss
from openrlhf.models.utils import log_probs_from_logits
from openrlhf.utils.distributed_sampler import DistributedSampler

import jsonlines
import os
import json
from openrlhf.models import MetaDPOLoss
from openrlhf.trainer import DPOTrainer


class MetaDPOTrainer(DPOTrainer):
    """
    Trainer for Meta DPO training that alternates between training the model and meta learner.
    """
    def __init__(
        self,
        model,
        ref_model,
        strategy,
        tokenizer,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm=0.5,
        beta=0.01,
        max_epochs: int = 2,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        meta_learner=None,
        meta_optim=None,
    ) -> None:
        super().__init__(
            model,
            ref_model,
            strategy,
            tokenizer,
            optim,
            train_dataloader,
            eval_dataloader,
            scheduler,
            max_norm,
            beta,
            max_epochs,
            save_hf_ckpt,
            disable_ds_ckpt,
        )
        self.meta_learner = meta_learner
        self.meta_optimizer = meta_optim
        
        # Replace DPO loss with Meta DPO loss
        self.loss_fn = MetaDPOLoss(
            self.beta,
            self.meta_learner,
            self.args.label_smoothing,
            self.args.loss_type,
            self.args.no_meta_weight,
        )
        
        self.meta_k = self.args.meta_k
        
        # 添加缓存来存储 meta learner 的训练数据
        self.meta_train_buffer = {
            'chosen_logps': [],
            'rejected_logps': [],
            'reference_chosen_logps': [],
            'reference_rejected_logps': [],
            'online_chosen_logps': [],
            'online_rejected_logps': [],
            'online_reference_chosen_logps': [],
            'online_reference_rejected_logps': []
        }
        
        self.meta_accmulate_steps = self.args.train_batch_size // (self.args.micro_train_batch_size * self.strategy.world_size)
        self.meta_buffer_size = (self.args.train_batch_size * self.meta_k) // self.strategy.world_size
        self.meta_save_path = os.path.join(self.args.save_path, "meta_learner")
        os.makedirs(self.meta_save_path, exist_ok=True)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        acc_sum = 0
        loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if not self.args.loss_type == "simpo":
                self.ref_model.eval()
            self.meta_learner.eval()
            # train
            for data in self.train_dataloader:
                if not self.packing_samples:
                    (chosen_ids, c_mask, reject_ids, r_mask, 
                     online_chosen_ids, online_c_mask, online_reject_ids, online_r_mask, 
                     prompt_id_lens) = data
                    
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    
                    if online_chosen_ids is not None:
                        online_chosen_ids = online_chosen_ids.squeeze(1).to(torch.cuda.current_device())
                        online_c_mask = online_c_mask.squeeze(1).to(torch.cuda.current_device())
                        online_reject_ids = online_reject_ids.squeeze(1).to(torch.cuda.current_device())
                        online_r_mask = online_r_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_logps, rejected_logps, aux_loss, nll_loss = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    with torch.no_grad():
                        if self.args.loss_type == "simpo":
                            reference_chosen_logps, reference_rejected_logps = None, None
                        else:
                            reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                                self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                            )
                    
                    # Process online data forward propagation
                    if online_chosen_ids is not None:
                        online_chosen_logps, online_rejected_logps, online_aux_loss, online_nll_loss = self.concatenated_forward(
                            self.model, online_chosen_ids, online_c_mask, online_reject_ids, online_r_mask, prompt_id_lens
                        )
                        with torch.no_grad():
                            if self.args.loss_type == "simpo":
                                online_reference_chosen_logps, online_reference_rejected_logps = None, None
                            else:
                                online_reference_chosen_logps, online_reference_rejected_logps, _, _ = self.concatenated_forward(
                                    self.ref_model, online_chosen_ids, online_c_mask, online_reject_ids, online_r_mask, prompt_id_lens
                                )
                    else:
                        online_chosen_logps = online_rejected_logps = None
                        online_reference_chosen_logps = online_reference_rejected_logps = None
                        online_aux_loss = online_nll_loss = 0
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())
                    
                    # 使用相同的 packed_samples_forward 函数处理所有数据
                    chosen_logps, rejected_logps, online_chosen_logps, online_rejected_logps, aux_loss, nll_loss = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                    )
                    with torch.no_grad():
                        if self.args.loss_type == "simpo":
                            reference_chosen_logps, reference_rejected_logps = None, None
                            online_reference_chosen_logps = online_reference_rejected_logps = None
                        else:
                            reference_chosen_logps, reference_rejected_logps, online_reference_chosen_logps, online_reference_rejected_logps, _, _ = self.packed_samples_forward(
                                self.ref_model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                            )

                # Update loss function call, including online data
                preference_loss, chosen_reward, reject_reward, online_chosen_reward, online_reject_reward = self.loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps,
                    online_chosen_logps, online_rejected_logps,
                    online_reference_chosen_logps, online_reference_rejected_logps
                )
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0
                # nll loss
                if not self.nll_loss:
                    nll_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef + nll_loss * self.args.nll_loss_coef

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # Calculate online accuracy
                acc = (chosen_reward > reject_reward).float().mean().item()
                online_acc = 0
                if online_chosen_reward is not None and online_reject_reward is not None:
                    online_acc = (online_chosen_reward > online_reject_reward).float().mean().item()
                
                acc_sum += acc
                loss_sum += preference_loss.item()
                
                # dpo logs
                logs_dict = {
                    "loss": preference_loss.item(),
                    "acc": acc,
                    "online_acc": online_acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                
                if online_chosen_reward is not None:
                    logs_dict["online_chosen_reward"] = online_chosen_reward.mean().item()
                    logs_dict["online_reject_reward"] = online_reject_reward.mean().item()
                    
                if self.nll_loss:
                    logs_dict["nll_loss"] = nll_loss.item()
                
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # Add current batch data to meta train buffer
                if not self.args.no_meta_weight:
                    with torch.no_grad():
                        self.meta_train_buffer['chosen_logps'].append(chosen_logps.detach())
                        self.meta_train_buffer['rejected_logps'].append(rejected_logps.detach())
                        if self.args.loss_type == "simpo":
                            self.meta_train_buffer['reference_chosen_logps'].append(None)
                            self.meta_train_buffer['reference_rejected_logps'].append(None)
                        else:
                            self.meta_train_buffer['reference_chosen_logps'].append(reference_chosen_logps.detach())
                            self.meta_train_buffer['reference_rejected_logps'].append(reference_rejected_logps.detach())
                        
                        if online_chosen_logps is not None:
                            self.meta_train_buffer['online_chosen_logps'].append(online_chosen_logps.detach())
                            self.meta_train_buffer['online_rejected_logps'].append(online_rejected_logps.detach())
                            if self.args.loss_type == "simpo":
                                self.meta_train_buffer['online_reference_chosen_logps'].append(None)
                                self.meta_train_buffer['online_reference_rejected_logps'].append(None)
                            else:
                                self.meta_train_buffer['online_reference_chosen_logps'].append(online_reference_chosen_logps.detach())
                                self.meta_train_buffer['online_reference_rejected_logps'].append(online_reference_rejected_logps.detach())
                        else:
                            self.meta_train_buffer['online_chosen_logps'].append(None)
                            self.meta_train_buffer['online_rejected_logps'].append(None)
                            self.meta_train_buffer['online_reference_chosen_logps'].append(None)
                            self.meta_train_buffer['online_reference_rejected_logps'].append(None)

                    # When the cache reaches the specified size, train meta learner
                    if len(self.meta_train_buffer['chosen_logps']) >= self.meta_buffer_size:
                        meta_loss, meta_acc, meta_online_acc = self.train_meta_step()
                        logs_dict["meta_loss"] = meta_loss
                        logs_dict["meta_acc"] = meta_acc
                        logs_dict["meta_online_acc"] = meta_online_acc
                        
                        # Save meta learner every fixed number of steps
                        if step % (4 * self.meta_k * self.strategy.accumulated_gradient) == 0:
                            self.save_meta_learner(step)
                        
                        # Clear cache
                        self.meta_train_buffer = {k: [] for k in self.meta_train_buffer.keys()}

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    acc_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        self.save_meta_learner("latest", save_model=True)
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


    def train_meta_step(self):
        """Train meta learner using accumulated data"""
        self.model.eval()
        self.meta_learner.train()
        
        self.strategy.print(f"Starting meta training step with buffer size {len(self.meta_train_buffer['chosen_logps'])}...")
        
        loss_sum = 0
        meta_acc_sum = 0
        meta_online_acc_sum = 0
        
        # 将 self.meta_train_buffer 分成 self.meta_accmulate_steps 个批次
        micro_batch_size = len(self.meta_train_buffer['chosen_logps']) // self.meta_accmulate_steps
        for i in range(self.meta_accmulate_steps):
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size if i < self.meta_accmulate_steps - 1 else len(self.meta_train_buffer['chosen_logps'])
            batch_train_data = {
                'chosen_logps': self.meta_train_buffer['chosen_logps'][start_idx:end_idx],
                'rejected_logps': self.meta_train_buffer['rejected_logps'][start_idx:end_idx],
                'reference_chosen_logps': self.meta_train_buffer['reference_chosen_logps'][start_idx:end_idx],
                'reference_rejected_logps': self.meta_train_buffer['reference_rejected_logps'][start_idx:end_idx],
                'online_chosen_logps': self.meta_train_buffer['online_chosen_logps'][start_idx:end_idx],
                'online_rejected_logps': self.meta_train_buffer['online_rejected_logps'][start_idx:end_idx],
                'online_reference_chosen_logps': self.meta_train_buffer['online_reference_chosen_logps'][start_idx:end_idx],
                'online_reference_rejected_logps': self.meta_train_buffer['online_reference_rejected_logps'][start_idx:end_idx],
            }
            
            chosen_logps = torch.cat(batch_train_data['chosen_logps'], dim=0)
            rejected_logps = torch.cat(batch_train_data['rejected_logps'], dim=0)
            if self.args.loss_type == "simpo":
                reference_chosen_logps = None
                reference_rejected_logps = None
            else:
                reference_chosen_logps = torch.cat(batch_train_data['reference_chosen_logps'], dim=0)
                reference_rejected_logps = torch.cat(batch_train_data['reference_rejected_logps'], dim=0)
            
            # 处理 online 数据
            online_data_available = all(x is not None for x in batch_train_data['online_chosen_logps'])
            if online_data_available:
                online_chosen_logps = torch.cat(batch_train_data['online_chosen_logps'], dim=0)
                online_rejected_logps = torch.cat(batch_train_data['online_rejected_logps'], dim=0)
                if self.args.loss_type == "simpo":
                    online_reference_chosen_logps = None
                    online_reference_rejected_logps = None
                else:
                    online_reference_chosen_logps = torch.cat(batch_train_data['online_reference_chosen_logps'], dim=0)
                    online_reference_rejected_logps = torch.cat(batch_train_data['online_reference_rejected_logps'], dim=0)
            else:
                online_chosen_logps = online_rejected_logps = None
                online_reference_chosen_logps = online_reference_rejected_logps = None

            # Train meta learner
            loss, chosen_reward, reject_reward, online_chosen_reward, online_reject_reward = self.loss_fn(
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps,
                online_chosen_logps, online_rejected_logps,
                online_reference_chosen_logps, online_reference_rejected_logps,
                train_meta=True
            )
            
            # 计算 accuracy
            meta_acc = (chosen_reward > reject_reward).float().mean().item()
            meta_online_acc = 0
            if online_data_available:
                meta_online_acc = (online_chosen_reward > online_reject_reward).float().mean().item()
            
            self.strategy.backward(loss, self.meta_learner, self.meta_optimizer)
            self.strategy.optimizer_step(self.meta_optimizer, self.meta_learner, None)
            
            loss_sum += loss.item()
            meta_acc_sum += meta_acc
            meta_online_acc_sum += meta_online_acc
            
        self.model.train()
        self.meta_learner.eval()
        mean_loss = loss_sum / self.meta_accmulate_steps
        mean_meta_acc = meta_acc_sum / self.meta_accmulate_steps
        mean_meta_online_acc = meta_online_acc_sum / self.meta_accmulate_steps
        
        self.strategy.print(f"Meta training step completed, accumulated steps: {self.meta_accmulate_steps}")
        self.strategy.print(f"Meta loss: {mean_loss:.4f}")
        self.strategy.print(f"Meta acc: {mean_meta_acc:.4f}")
        self.strategy.print(f"Meta online acc: {mean_meta_online_acc:.4f}")
        return mean_loss, mean_meta_acc, mean_meta_online_acc
    
    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            acc_sum = 0
            loss_sum = 0
            times = 0
            for data in eval_dataloader:
                if not self.packing_samples:
                    (chosen_ids, c_mask, reject_ids, r_mask, 
                     online_chosen_ids, online_c_mask, online_reject_ids, online_r_mask, 
                     prompt_id_lens) = data
                    
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    if online_chosen_ids is not None:
                        online_chosen_ids = online_chosen_ids.squeeze(1).to(torch.cuda.current_device())
                        online_c_mask = online_c_mask.squeeze(1).to(torch.cuda.current_device())
                        online_reject_ids = online_reject_ids.squeeze(1).to(torch.cuda.current_device())
                        online_r_mask = online_r_mask.squeeze(1).to(torch.cuda.current_device())
                    else:
                        online_chosen_ids = None
                        online_c_mask = None
                        online_reject_ids = None
                        online_r_mask = None

                    chosen_logps, rejected_logps, aux_loss, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    with torch.no_grad():
                        if self.args.loss_type == "simpo":
                            reference_chosen_logps, reference_rejected_logps = None, None
                        else:
                            reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                                self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                            )
                    if online_chosen_ids is not None:
                        online_chosen_logps, online_rejected_logps, _, _ = self.concatenated_forward(
                            self.model, online_chosen_ids, online_c_mask, online_reject_ids, online_r_mask, prompt_id_lens
                        )
                        online_reference_chosen_logps, online_reference_rejected_logps, _, _ = self.concatenated_forward(
                            self.ref_model, online_chosen_ids, online_c_mask, online_reject_ids, online_r_mask, prompt_id_lens
                        )
                    else:
                        online_chosen_logps = online_rejected_logps = None
                        online_reference_chosen_logps = online_reference_rejected_logps = None
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())
                    chosen_logps, rejected_logps, online_chosen_logps, online_rejected_logps, aux_loss, _ = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                    )
                    with torch.no_grad():
                        if self.args.loss_type == "simpo":
                            reference_chosen_logps, reference_rejected_logps = None, None
                            online_reference_chosen_logps = online_reference_rejected_logps = None
                        else:
                            reference_chosen_logps, reference_rejected_logps, online_reference_chosen_logps, online_reference_rejected_logps, _, _ = self.packed_samples_forward(
                                self.ref_model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                            )

                loss, chosen_reward, reject_reward, online_chosen_reward, online_reject_reward = self.loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps,
                    online_chosen_logps, online_rejected_logps,
                    online_reference_chosen_logps, online_reference_rejected_logps
                )
                acc_sum += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                times += 1
                step_bar.update()

            logs = {
                "eval_loss": loss_sum / times,
                "acc_mean": acc_sum / times,
            }
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state
        
    def save_meta_learner(self, step, save_model=False):
        """Save meta learner checkpoint (DeepSpeed)"""
        save_path = os.path.join(self.meta_save_path)
        if self.strategy.is_rank_0():
            os.makedirs(save_path, exist_ok=True)
            # DeepSpeed engine save
        torch.distributed.barrier()
        self.meta_learner.save_checkpoint(save_path, tag=f"step_{step}")
        self.strategy.print(f"Saved meta learner checkpoint to {save_path}, tag step_{step}")
        
        if save_model and self.strategy.is_rank_0():
            meta_state = {"model_state_dict": self.meta_learner.state_dict()}
            torch.save(meta_state, os.path.join(save_path, "meta_learner.pt"))
            
    def load_meta_learner(self, path):
        """Load meta learner checkpoint"""
        if not os.path.exists(path):
            self.strategy.print(f"No meta learner checkpoint found at {path}")
            return
        
        meta_state = torch.load(path, map_location=torch.cuda.current_device())
        
        self.meta_learner.load_state_dict(meta_state['model_state_dict'])
        self.meta_optimizer.load_state_dict(meta_state['optimizer_state_dict'])
        
        self.strategy.print(f"Loaded meta learner checkpoint from {path}")
        return meta_state['step']
    
    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens):
        output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._packed_get_batch_logps(
            all_logits,
            packed_input_ids,
            packed_attention_masks,
            prompt_id_lens * 4,
            packed_seq_lens,
            average_log_prob=False,
        )
        
        # Split the log probabilities into four parts
        quarter_point = len(packed_seq_lens) // 4
        chosen_logps = all_logps_sum[:quarter_point]
        rejected_logps = all_logps_sum[quarter_point:quarter_point*2]
        online_chosen_logps = all_logps_sum[quarter_point*2:quarter_point*3]
        online_rejected_logps = all_logps_sum[quarter_point*3:]
        
        aux_loss = output.aux_loss if "aux_loss" in output else []
        nll_loss = -all_logps_mean[:quarter_point].mean()  # Only compute NLL loss on chosen responses
        
        return chosen_logps, rejected_logps, online_chosen_logps, online_rejected_logps, aux_loss, nll_loss
