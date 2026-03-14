# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)



    def _forward_micro_batch(self, micro_batch, temperature, advantages=None) -> dict:
        """
        Args:
            micro_batch: Batch data
            temperature: Sampling temperature
            advantages: Optional per-token advantages [B, response_length]. (not used here)

        Returns:
            dict:
                log_probs: (bs, response_len)
                entropys: (bs, response_len)   # always computed/returned
                log_probs_others: (bs, response_len) or (bs, response_len, k), optional
                yprimes: (bs, response_len) or (bs, response_len, k), optional
        """
        import torch

        positive_transform = bool(self.config.get("positive_transform", False))
        num_yprimes = max(1, int(self.config.get("positive_transform_num_yprimes", 1)))

        response_length = micro_batch["responses"].size(-1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if "loss_mask" in micro_batch:
                response_mask = micro_batch["loss_mask"]
            else:
                response_mask = attention_mask[:, -response_length:]

            entropy = None
            log_probs_others = None
            yprimes = None

            if self.use_remove_padding:
                # rmpad
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad position_ids
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # mask for logits positions that predict response tokens
                response_logit_mask_full = torch.zeros(
                    batch_size, seqlen, device=attention_mask.device, dtype=torch.bool
                )
                response_logit_mask_full[:, -response_length - 1 : -1] = response_mask.bool()
                response_logit_mask_rmpad = index_first_axis(
                    rearrange(response_logit_mask_full.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).squeeze(-1)  # [total_nnz] bool

                # next-token labels (rmpad space)
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # ulysses sp pad/slice
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                    )
                    response_logit_mask_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                        response_logit_mask_rmpad.unsqueeze(0), None, self.ulysses_sequence_parallel_size
                    )
                    response_logit_mask_rmpad = response_logit_mask_rmpad.squeeze(0).bool()

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # [nnz]

                # forward varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)  # [nnz, V]
                logits_rmpad.div_(temperature)

                # log_prob(y)
                log_probs_rmpad = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                )  # [nnz]

                # entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # [nnz]

                # positive_transform: compute / reuse yprimes + log_probs_others on response positions
                if positive_transform:
                    logits_resp = logits_rmpad[response_logit_mask_rmpad]      # [Nresp, V]
                    y_resp = input_ids_rmpad_rolled[response_logit_mask_rmpad] # [Nresp]
                    V = logits_resp.size(-1)

                    if V > 1 and logits_resp.numel() > 0:
                        if "yprimes" in micro_batch:
                            # reuse provided yprimes [B,R] or [B,R,K], aligned with logits slice [-R-1:-1]
                            yprimes_br = micro_batch["yprimes"].to(device=input_ids.device, dtype=torch.long)
                            if yprimes_br.ndim == 2:
                                yprimes_br = yprimes_br.unsqueeze(-1)
                            K = yprimes_br.size(-1)

                            yprime_pred_full = torch.zeros(
                                batch_size, seqlen, K, device=input_ids.device, dtype=torch.long
                            )
                            yprime_pred_full[:, -response_length - 1 : -1, :] = yprimes_br

                            yprime_pred_rmpad = index_first_axis(
                                rearrange(yprime_pred_full, "b s k -> (b s) k"),
                                indices
                            )  # [total_nnz, K]

                            if self.use_ulysses_sp:
                                yprime_pred_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                                    yprime_pred_rmpad.unsqueeze(0), None, self.ulysses_sequence_parallel_size
                                )
                                yprime_pred_rmpad = yprime_pred_rmpad.squeeze(0)

                            yprime_resp = yprime_pred_rmpad[response_logit_mask_rmpad]  # [Nresp, K]
                        else:
                            probs = torch.softmax(logits_resp.float(), dim=-1)  # [Nresp, V]
                            probs.scatter_(1, y_resp.unsqueeze(1), 0.0)
                            denom = probs.sum(dim=-1, keepdim=True)
                            bad = denom.squeeze(-1) <= 0.0
                            if bad.any():
                                probs[bad].fill_(1.0)
                                probs[bad].scatter_(1, y_resp[bad].unsqueeze(1), 0.0)
                                denom = probs.sum(dim=-1, keepdim=True)
                            probs = probs / denom.clamp_min(1e-12)
                            yprime_resp = torch.multinomial(
                                probs, num_samples=num_yprimes, replacement=True
                            ).to(dtype=torch.long)  # [Nresp, K]

                        Nresp = logits_resp.size(0)
                        K = yprime_resp.size(-1)
                        logits_resp_expanded = logits_resp.unsqueeze(1).expand(Nresp, K, V).reshape(-1, V)
                        logp_other_resp = logprobs_from_logits(
                            logits=logits_resp_expanded,
                            labels=yprime_resp.reshape(-1),
                        ).view(Nresp, K)  # [Nresp, K]
                    else:
                        K = num_yprimes
                        yprime_resp = y_resp.to(dtype=torch.long).unsqueeze(-1).expand(-1, K)  # [Nresp, K]
                        logp_other_resp = torch.zeros(
                            yprime_resp.shape, device=logits_resp.device, dtype=log_probs_rmpad.dtype
                        )

                    nnz = logits_rmpad.size(0)
                    K = yprime_resp.size(-1)
                    logp_other_rmpad = log_probs_rmpad.new_zeros((nnz, K))
                    yprime_rmpad = input_ids_rmpad_rolled.new_zeros((nnz, K), dtype=torch.long)
                    logp_other_rmpad[response_logit_mask_rmpad] = logp_other_resp
                    yprime_rmpad[response_logit_mask_rmpad] = yprime_resp.to(dtype=torch.long)

                # gather if ulysses sp
                if self.use_ulysses_sp:
                    log_probs_rmpad = gather_outpus_and_unpad(
                        log_probs_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    entropy_rmpad = gather_outpus_and_unpad(
                        entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    if positive_transform:
                        logp_other_rmpad = gather_outpus_and_unpad(
                            logp_other_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                        yprime_rmpad = gather_outpus_and_unpad(
                            yprime_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                # pad back to [B, seqlen]
                full_entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_log_probs = pad_input(
                    hidden_states=log_probs_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]        # [B, R]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]    # [B, R]

                if positive_transform:
                    full_logp_other = pad_input(
                        hidden_states=logp_other_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_yprime = pad_input(
                        hidden_states=yprime_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    log_probs_others = full_logp_other[:, -response_length - 1 : -1, :]  # [B, R, K]
                    yprimes = full_yprime[:, -response_length - 1 : -1, :]               # [B, R, K]

            else:
                # dense path
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1]  # [B, R, V]

                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                entropy = verl_F.entropy_from_logits(logits)

                if positive_transform:
                    y = micro_batch["responses"]  # [B, R]
                    B, R, V = logits.shape

                    if "yprimes" in micro_batch:
                        yprimes = micro_batch["yprimes"].to(device=logits.device, dtype=y.dtype)
                        if yprimes.ndim == 2:
                            yprimes = yprimes.unsqueeze(-1)
                    else:
                        if V > 1:
                            probs = torch.softmax(logits.float(), dim=-1)  # [B,R,V]
                            probs.scatter_(-1, y.unsqueeze(-1), 0.0)
                            denom = probs.sum(dim=-1, keepdim=True)
                            bad = denom.squeeze(-1) <= 0.0
                            if bad.any():
                                probs[bad].fill_(1.0)
                                probs[bad].scatter_(-1, y[bad].unsqueeze(-1), 0.0)
                                denom = probs.sum(dim=-1, keepdim=True)
                            probs = probs / denom.clamp_min(1e-12)

                            yprimes = torch.multinomial(
                                probs.view(-1, V), num_samples=num_yprimes, replacement=True
                            ).view(B, R, num_yprimes).to(dtype=y.dtype)
                        else:
                            yprimes = y.unsqueeze(-1).expand(B, R, num_yprimes).clone()

                    K = yprimes.size(-1)
                    logits_expanded = logits.unsqueeze(2).expand(B, R, K, V).reshape(-1, V)
                    log_probs_others = logprobs_from_logits(
                        logits_expanded,
                        yprimes.reshape(-1),
                    ).view(B, R, K)

            if positive_transform and log_probs_others is not None and log_probs_others.dim() == 3 and log_probs_others.size(-1) == 1:
                log_probs_others = log_probs_others.squeeze(-1)
                yprimes = yprimes.squeeze(-1)

            outputs = {
                "log_probs": log_probs,
                "entropys": entropy,
                "log_probs_others": log_probs_others,
                "yprimes": yprimes,
            }
            return outputs


    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> dict:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Returns:
            dict:
                - log_probs: [B, R]
                - entropys:  [B, R]
                - log_probs_others: [B, R] (optional, if positive_transform)
                - yprimes: [B, R] (optional, if positive_transform)
        """
        import itertools

        positive_transform = bool(self.config.get("positive_transform", False))

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropys_lst = []

        # NEW: only if positive_transform
        log_probs_others_lst = []
        yprimes_lst = []

        for micro_batch in micro_batches:
            with torch.no_grad():
                outputs = self._forward_micro_batch(micro_batch, temperature=temperature, advantages=None)

            # Keep recomputed log-prob tensors on CPU to reduce GPU memory pressure.
            log_probs_lst.append(outputs["log_probs"].detach().to("cpu"))
            entropys_lst.append(outputs["entropys"].detach().to("cpu"))

            if positive_transform:
                # Keep positive-transform aux tensors on CPU during accumulation to reduce GPU peak memory.
                log_probs_others_lst.append(outputs["log_probs_others"].detach().to("cpu"))
                yprimes_lst.append(outputs["yprimes"].detach().to("cpu"))

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropys_lst, dim=0)

        if positive_transform:
            log_probs_others = torch.concat(log_probs_others_lst, dim=0)
            yprimes = torch.concat(yprimes_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=log_probs.device)

            log_probs = log_probs[revert_indices]
            entropys = entropys[revert_indices]
            if positive_transform:
                revert_indices_cpu = revert_indices.to("cpu")
                log_probs_others = log_probs_others[revert_indices_cpu]
                yprimes = yprimes[revert_indices_cpu]

        outputs = {"log_probs": log_probs, "entropys": entropys}
        if positive_transform:
            outputs["log_probs_others"] = log_probs_others
            outputs["yprimes"] = yprimes

        return outputs

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']

        positive_transform = bool(self.config.get("positive_transform", False))

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        # compute once under pi_old BEFORE any update
        if positive_transform:
            select_keys.append('old_log_prob_others')
            select_keys.append('yprimes')

        batch = data.select(batch_keys=select_keys).batch


        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for idx, mini_batch in enumerate(dataloader):
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for mb in micro_batches:
                mb = mb.cuda()

                responses = mb['responses']
                R = responses.size(1)
                attention_mask = mb['attention_mask']
                response_mask = attention_mask[:, -R:]
                if self.config.state_masking:
                    response_mask = mb['loss_mask']

                old_log_prob = mb['old_log_probs']
                advantages = mb['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff
                positive_transform_beta = float(self.config.get("positive_transform_beta", 0.0))

                outputs = self._forward_micro_batch(micro_batch=mb, temperature=temperature, advantages=advantages)
                log_prob = outputs["log_probs"]
                entropy = outputs["entropys"]

                if positive_transform:
                    old_log_prob_other = mb["old_log_prob_others"].to(log_prob.device, non_blocking=True)
                    log_prob_other = outputs["log_probs_others"]
                    pg_loss, pg_clipfrac, w_clipfrac, ppo_kl = core_algos.compute_policy_loss_positive_negative_trans(
                        old_log_prob_y=old_log_prob,
                        log_prob_y=log_prob,
                        old_log_prob_other=old_log_prob_other,
                        log_prob_other=log_prob_other,
                        advantages=advantages,
                        eos_mask=response_mask,
                        cliprange=clip_ratio,
                        beta=positive_transform_beta,
                    )
                else:
                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        eos_mask=response_mask,
                        cliprange=clip_ratio,
                    )

                entropy_loss = verl_F.masked_mean(entropy, response_mask)
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                # KL stays original: always uses log_prob(y)
                if self.config.use_kl_loss:
                    ref_log_prob = mb['ref_log_prob']
                    kld = core_algos.kl_penalty(
                        logprob=log_prob,
                        ref_logprob=ref_log_prob,
                        kl_penalty=self.config.kl_loss_type
                    )
                    kl_loss = masked_mean(kld, response_mask)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                (policy_loss / self.gradient_accumulation).backward()

                append_to_dict(metrics, {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                })
                if positive_transform:
                    append_to_dict(metrics, {
                        'actor/w_clipfrac': w_clipfrac.detach().item(),
                    })                    

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
