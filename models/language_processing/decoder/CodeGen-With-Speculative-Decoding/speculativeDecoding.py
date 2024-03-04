##############################################################################
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
##############################################################################

import json
import os
from time import perf_counter
from typing import Dict, List, Optional
from pdb import set_trace as bp
import numpy as np
import transformers
from transformers import (
    TopKLogitsWarper,
    LogitsWarper,
)
import torch
torch.manual_seed(42)
np.random.seed(42)
from qaic_infer import QAICInferenceSession

from rich.console import Console

console = Console()

DEBUG = 1
UNDERSCORE_SPECIAL_TOKEN = 9601
def debugprint(*args):
    global DEBUG
    if False:
        print(args)

time_dict = dict()
n_tokens_accepted_histogram = dict()
def update_time(key, start_time):
    global time_dict
    if DEBUG == 0:
        return
    if key not in time_dict.keys():
        time_dict[key] = 0
    time_dict[key] += (perf_counter() - start_time)

class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        # NOTE: 1st input argument was modified from 1 in HF's TopPLogitsWarper to -1
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

    def get_logits_and_indices_to_remove(self, logits: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        # NOTE: 1st input argument was modified from 1 in HF's TopPLogitsWarper to -1
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, self.filter_value)
        return logits, indices_to_remove

io_files = []

vocab_size_dict = {"llama2":32000, "codegen":51200}

#TODO remove this hard-coding
max_spec_length = 7
max_warped_tokens = 0
to_use_gumbel = False
#1/11/24:  We can also achieve greedy by top_k=1.  Leaving these two variables in for now,
#but after discussion, we should not need to change these two parameters ever to False.

#The script assumes that if --exact-greedy is passed, we perform top-1 warping of both
# TLM and DLM logits, sample from them and perform MRS using the warped distributions
# consistent tiebreaking is ensured as well (selects the first index from the filtered top-1 indices)
draft_to_sample = True
target_to_sample = True

stop_sequences = [
        "\n\n\n"
      ]

def get_warped_logits(logits_np, top_k, top_p, topk_logit_warper, topp_logit_warper, axis=-1, temperature=1.0, max_num_tokens=0):
    # creating a float32 copy of the input logits to run log_softmax on

    logits = torch.tensor(np.array(logits_np, dtype=np.float32))
    # Top-k sampling
    if top_k > 0:
        logits = topk_logit_warper(scores=logits, input_ids=None)
        #agokhale tiebreaker in case of multiple indices with the same largest value
        if max_num_tokens > 0:
            #we only want the first max_num_tokens that are filtered out by top-k sampling
            filtered_indices = torch.nonzero(logits != - float("Inf"))[:,-1]
            num_tokens_filtered = filtered_indices.shape[0]
            if num_tokens_filtered > max_num_tokens:
                debugprint("[WARN] number of tokens that were filtered out > max_num_tokens; masking out all indices except the first max_num_tokens")
                logits[...,filtered_indices[max_num_tokens:]] = - float("Inf")

    # Top-p sampling 
    if top_p < 1:
        logits = topp_logit_warper(scores=logits, input_ids=None)
    logits = logits / temperature
    return logits.numpy()

def get_gumbel(size: int):
    tiny = np.float64(1.1754943508222875e-38)
    eps = np.float64(1.1920928955078125e-07)
    low = tiny
    high = 1. - eps
    
    u = np.random.rand(size)
    u = low + u * (high - low)
    gumbel = - np.log(- np.log(u))
    return gumbel

#Was not sure if this was numerically equivalent/stable
#calculates log_softmax but in numpy domain not tensor domain
#def log_softmax(x):
#    e_x = np.exp(x - np.max(x))
#    return np.log(e_x / e_x.sum())

def sample_from_logits(logits_np, n_samples: int = 1, do_sample: bool = True, use_gumbel: bool = True):
    # returns sampled tokens in the form of a numpy array
    #print("Shape of Logits:", logits_np.shape)
    if not isinstance(logits_np, torch.Tensor):
        logits = torch.tensor(logits_np)
    else:
        logits = logits_np
    retval = None
    if use_gumbel:
        if do_sample:#gumbel sampling
            gumbel = get_gumbel(logits.numel())
            logits = logits + gumbel
        if n_samples > 1:
            return torch.topk(logits, n_samples, dim=-1).indices.flatten(start_dim=1)
        else:
            retval = torch.argmax(logits, dim=-1)
    else:
        if do_sample:#multinomial sampling
            # Modified to return tokens in the same dimensionality as gumbel
            # Using logits[0] since torch.multinomial only accepts 1-d/2-d tensors while logits is a 3-d tensor for draft sampling
            retval = torch.multinomial(torch.nn.functional.softmax(logits[0], dim=-1), 1, replacement=False).flatten().view(1, -1)
        else:#greedy
            if n_samples > 1:
                return torch.topk(logits, n_samples, dim=-1).indices.flatten(start_dim=1)
            else:
                retval = torch.argmax(logits, dim=-1)
    if len(retval.shape) < 2:
        retval = torch.unsqueeze(retval, axis=1)
    debugprint("returning token value:", retval.numpy())
    return retval.numpy()

def rejection_sampling(token, logp_draft_np, logp_target_np, do_sample_draft, do_sample_main, use_gumbel, target_token, sample_matching):

        # returns: bool whether current token is accepted or not
        # and numpy array holding the correct token from TLM
        logp_draft = torch.tensor(logp_draft_np.copy())
        logp_target = torch.tensor(logp_target_np.copy())
        if sample_matching:
            acceptance_criteria = token == target_token
        else:
            if do_sample_main:
                logp_target_threshold = logp_target[:,token]
            else:
                if token == target_token:
                    logp_target_threshold = 0.              #reflects greediness of TLM
                else:
                    logp_target_threshold = - float('inf')  #threshold will be 0, never accept. Fine.

            if do_sample_draft:
                logp_draft_threshold = logp_draft[:,token]  #get probability from PDF.
            else:
                logp_draft_threshold = 0.                   #greedy... shouldnt i make threshold=1

            # sample main, sample draft:
            #  main uses pdf, draft uses pdf.  if draft bigger pdf[token], sometimes reject. OK.
            # sample main, greedy draft:
            #  main uses pdf, draft uses 0 value.   0 means its the biggest.  Ah ok, greedy.  so, redistribute. OK.
            # greedy main, sample draft:
            #  main threshold is either 0 or -inf, so reverts back to matching or not.  kind of unneccessarily underperforming, why sample draft then?
            # greedy main, greedy draft:
            #  main uses pdf, draft uses pdf.  reverts back to pure greedy matching or not.


            #if threshholds match, returns e^0=1.  if target bigger, e^0.3=1.3  if draft bigger, e^-0.3=0.7

            #ex: 0.3 0.2, yields 1.5...ratio of the two.
            threshold = (logp_target_threshold - logp_draft_threshold).clone().exp()
            debugprint(f"value of threshold: {threshold}")
            #between 0 and 1.
            u = np.random.rand()
            #So if target is greater or equal than draft, threshold>=1, always accepted
            acceptance_criteria = torch.tensor(u, device=threshold.device) < threshold
            debugprint(f"result of mrs: {acceptance_criteria}")
        if acceptance_criteria:  # Acceptance
            accepted = True
            token_last = None
        else:  # Recursion or Rejection
            accepted = False
            if do_sample_main and not sample_matching:
                if do_sample_draft:
                    # Both target and draft in sampling mode
                    tiny = 1.1754943508222875e-38
                    logp_target = torch.clamp(logp_target.exp() - logp_draft.exp(), min=tiny).log()
                else:
                    # Target in sampling mode, draft in greedy mode
                    # In this case (target-draft)_+ = target with renormalization except at the greedy draft token where it will be 0
                    logp_target[:,token] = - float('inf')
                token_last = sample_from_logits(logp_target.view(1, -1), do_sample=do_sample_main, use_gumbel=use_gumbel)
            else:
                #print("Returning the Target Token ",target_token)
                token_last = target_token
        return accepted, token_last

def main(
    model_family: str,
    model_name: str,
    prompt_len: int,
    ctx_len: int,
    dlm_qpc: str,
    tlm_qpc: str,
    prompt: List[str],
    max_spec_length: int,
    greedy: bool = False,
    stream: bool = True,
    device_id: List[int] = [0],
    #temperature_draft: float = 1.0,
    write_io_dir: Optional[str] = None,
) -> Dict[str, float]:
    global max_warped_tokens
    #defaults to MRS=true:
    is_exact_sample_matching = False
    # Part of the fix for when | pipe character is part of the prompt and not a separator
    prompt=[prompt]
    #apply temp,top-k,top-p in this sequence:
    temperature_target = 1
    temperature_draft = 1
    #set top_k=1 for greedy, =0 to disable
    draft_top_k = 0
    target_top_k = 0
    #set top_p=1 to disable
    draft_top_p = 1
    target_top_p = 1

    draft_topk_logit_warper = None
    draft_topp_logit_warper = None
    target_topk_logit_warper = None
    target_topp_logit_warper = None
    
    if greedy:
        draft_top_k = 1
        target_top_k = 1
        max_warped_tokens = 1

    # Initialize logit warper if using top-k or top-p sampling
    if draft_top_k > 0:
        draft_topk_logit_warper = TopKLogitsWarper(top_k=draft_top_k)
    if draft_top_p < 1:
        draft_topp_logit_warper = TopPLogitsWarper(top_p=draft_top_p)
    if target_top_k > 0:
        target_topk_logit_warper = TopKLogitsWarper(top_k=target_top_k)
    if target_top_p < 1:
        target_topp_logit_warper = TopPLogitsWarper(top_p=target_top_p)
    # Load QPC
    assert (model_family in vocab_size_dict.keys()), f"unsupported model family {model_family}"
    vocab_size = vocab_size_dict[model_family]

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left")
    dlm_session = QAICInferenceSession(f"{dlm_qpc}", device_id)
    dlm_session.skip_buffers(set([x for x in dlm_session.output_names if x.endswith("_RetainedState")]))
    tlm_session = QAICInferenceSession(f"{tlm_qpc}", device_id)
    tlm_session.skip_buffers(set([x for x in tlm_session.output_names if x.endswith("_RetainedState")]))
    #Why do we need to do this?  Should be unneccessary...
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare inputs for first iteration
    start = perf_counter()

    #run decode on DLM
    #initially, input_ids consist of the prompt, then subsequently they become either 1-long or 2-long vectors based on MRS outcome
    #position ids starts with the first token being assigned 0, and keeps increasing by 1 for each subsequent token
    #TODO attention mask: do we need to provide it anew for precode scenario?
    dlm_inputs = dict()
    tlm_inputs = dict()

    overall_num_tokens_accepted = 0
    dlm_cache_index = np.array([0])
    tlm_cache_index = np.array([0])
    batch_size = len(prompt)
    logits_out_placeholder = np.zeros((batch_size, vocab_size), dtype=np.float16)
    if model_family == "codegen":
        logits_out_placeholder = np.expand_dims(logits_out_placeholder, 1)
    #TODO handle batch_size>1 for prompt number of valid tokens
    prompt_num_valid_tokens = np.zeros((batch_size,1))
    #to store the final tokens accepted
    generated_ids = np.full((batch_size, ctx_len - prompt_len + 1), tokenizer.pad_token_id)
    num_dlm_tokens_rejected = -1
    #FIXME in the case of llama model vocabulary, the eos token == pad token. How to handle this?
    num_iters=0
    #store the new token generated by TLM in the previous iteration
    tlm_last_iter_newtoken_id = tokenizer.pad_token_id
    is_first_iter = True
    total_dlm_time = 0
    total_dlm_devrun_time = 0
    total_tlm_time = 0
    total_tlm_devrun_time = 0
    total_dlm_iter = 0
    total_tlm_iter = 0
    curr_dlm_time = 0
    curr_tlm_time = 0
    n_ttft = 0
    
    dlm_warped_logits = np.full((batch_size, max_spec_length, vocab_size), 0., dtype=np.float16)
    tlm_logits = np.full((batch_size, max_spec_length+1, vocab_size), 0., dtype=np.float16)
    tlm_warped_logits = np.full((batch_size, max_spec_length+1, vocab_size), 0., dtype=np.float16)
    dlm_session.set_buffers({"logits":logits_out_placeholder})
    tokens_accepted_in_first_round=0
    is_tlm_prefill = True
    # print out the prompt string
    if stream:
        console.print(f"[bold]{prompt[0] if isinstance(prompt, list) else prompt}", end="", highlight=False)
    print()
    print()
    while (overall_num_tokens_accepted + prompt_len) <  ctx_len:
        if n_ttft == 1:
          tokens_accepted_in_first_round = overall_num_tokens_accepted
          start_decode_throughput_counter = perf_counter()
        #only speculate up to the position id that's < maximum model context length
        stop_generation = False
        start_dlm = perf_counter()
        spec_length = min(max_spec_length, (ctx_len - prompt_len - overall_num_tokens_accepted))
        #to store the speculated tokens from DLM
        dlm_candidate_ids = np.full((batch_size, max_spec_length), tokenizer.pad_token_id)
        #to store the speculated token logits from DLM

        dlm_infer_start = perf_counter()
        for i in range(max_spec_length):
            dlm_preproc_start = perf_counter()
            if dlm_cache_index[0] == 0:#prefill
                dlm_inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=prompt_len)
                dlm_session.skip_buffers(set([x for x in dlm_session.input_names if x.startswith("past_")]))
                num_tokens = prompt_len
                dlm_inputs["position_ids"] = (np.cumsum(dlm_inputs["attention_mask"], 1) - 1) * dlm_inputs["attention_mask"]
                #FIXME the below line assumes batch size =1 and hence takes a full sum instead of along axis=1
                prompt_num_valid_tokens = np.sum(dlm_inputs["attention_mask"])
                dlm_inputs["attention_mask"] = np.concatenate(
                    [
                        dlm_inputs["attention_mask"].astype(bool),
                        np.zeros((batch_size, ctx_len - prompt_len), dtype=bool),
                    ],
                    1,
                )
                dlm_inputs["cache_index"] = dlm_cache_index

            else:#decode or precode
                dlm_inputs["attention_mask"] = np.zeros_like(dlm_inputs["attention_mask"])
                dlm_inputs["attention_mask"][0,prompt_len-prompt_num_valid_tokens:prompt_len+overall_num_tokens_accepted+i] = 1
                if num_dlm_tokens_rejected == 0:#run in precode mode for 2 input tokens
                    #no tokens were rejected in the previous TLM evaluation
                    # input has to be 2 tokens as well as their position_ids: (last_token_of_prev_speculation, new_token)
                    #reset the value of num_dlm_tokens_rejected to -1 to allow further single token decode speculation
                    num_dlm_tokens_rejected = -1
                    num_tokens = 2
                else:
                    num_tokens = 1
                    if len(next_token_id.shape) > 2:
                        next_token_id = np.squeeze(next_token_id, axis=-1)
                    dlm_inputs["input_ids"] = next_token_id
                    if dlm_cache_index[0] == prompt_len:
                        #previous iteration was a prefill
                        # Skip attention_mask from this iteration onwards to use retained attention_mask
                        dlm_inputs["position_ids"] = dlm_inputs["attention_mask"].sum(1, keepdims=True)
                    if dlm_inputs["position_ids"].shape[1] > 1:
                        #previous iteration was a precode
                        #next position id should be 1 more than that of the tlm_new_token
                        dlm_inputs["position_ids"] = dlm_inputs["position_ids"][:,1:2] + 1
                    else:
                        #previous iteration was a decode
                        dlm_inputs["position_ids"] += 1
            if dlm_cache_index[0] > 0:
                #only decode/precode time counted
                update_time("dlm_preproc", dlm_preproc_start)
            dlm_devrun_start = perf_counter()
            dlm_outputs = dlm_session.run(dlm_inputs)
            if dlm_cache_index[0] > 0:
                #only decode/precode time counted
                update_time("dlm_run", dlm_devrun_start)
            dlm_devrun_end=perf_counter()
            total_dlm_devrun_time += dlm_devrun_end-dlm_devrun_start
            total_dlm_iter += 1
            logits_out_dlm_warped = dlm_outputs["logits"].astype(np.float32)
            if greedy:
                # print("shape of logits_out_dlm_warped: ",logits_out_dlm_warped.shape)
                dlm_greedy_stime=perf_counter()
                next_token_id = np.expand_dims(np.argmax(logits_out_dlm_warped,axis=-1),0)
                update_time("dlm_sample", dlm_greedy_stime)
            else:
                ##Facing Judgement 1/3:
                if draft_to_sample:
                    draft_logit_warp_start = perf_counter()
                    logits_out_dlm_warped = get_warped_logits(logits_np=logits_out_dlm_warped, top_k=draft_top_k, top_p=draft_top_p,topk_logit_warper=draft_topk_logit_warper, topp_logit_warper=draft_topp_logit_warper, temperature=temperature_draft, max_num_tokens=max_warped_tokens)
                    update_time("draft_logit_warp", draft_logit_warp_start)
                dlm_warped_logits[:,i] = logits_out_dlm_warped[0]
                draft_sample_start = perf_counter()
                next_token_id = sample_from_logits(logits_np=logits_out_dlm_warped, do_sample=draft_to_sample, use_gumbel=to_use_gumbel)
                update_time("draft_sample", draft_sample_start)
            dlm_cache_index += num_tokens
            dlm_candidate_ids[:, i] = next_token_id
        dlm_infer_end = perf_counter()
        curr_dlm_time = dlm_infer_end - dlm_infer_start
        total_dlm_time = total_dlm_time + curr_dlm_time
        # generate candidate text from the DLM candidate ids to append to the prompt
        # dlm_candidate_texts = tokenizer.batch_decode(dlm_candidate_ids)
        # for idx in range(batch_size):
            # print(f"Candidate Text index {idx}: {dlm_candidate_texts[idx]}")

        tlm_infer_start = perf_counter()
        #print("\n **** TLM Inference ****")
        # in both cases, we want to extract the final spec_length + 1 logits
        # (1st iter prefill +)oneshot decode pass over the TLM 
        tlm_prefill_logit_out = None
        num_tlm_input_tokens = max_spec_length + 1 #for non-prefill scenario, we always provide the prev iteration's TLM token a followed by the speculation

        #if tlm_cache_index[0] != 0:
        if is_tlm_prefill == True:
            tlm_prefill_stime = perf_counter()
            #print("**TLM Prefill**")
            
            #is_tlm_prefill = True
            #tokenize the prompt, padding it to the prompt_len
            tlm_inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=prompt_len)
            #print("tlm_inputs\n")
            #print(tlm_inputs)
            # TODO fix for batch_size > 1, fix for when context-length-limit is reached
            #  when speculation can be smaller than spec_length(pad to the tlm_prompt_len)
            tlm_inputs["position_ids"] = (np.cumsum(tlm_inputs["attention_mask"], 1) - 1) * tlm_inputs["attention_mask"]
            tlm_session.skip_buffers(set([x for x in tlm_session.input_names if x.startswith("past_")]))
            tlm_inputs["attention_mask"] = np.concatenate(
                [
                    tlm_inputs["attention_mask"].astype(bool),
                    np.zeros((batch_size, ctx_len - prompt_len), dtype=bool),
                ],
                1,
            )
            num_tokens = prompt_len
            tlm_inputs["cache_index"] = tlm_cache_index # = 0 for prefill
            #print("TLM PREFILL ATTENTION MASK: ",tlm_inputs["attention_mask"].astype(int))
            #print("TLM PREFILL ATTENTION MASK length: ",np.sum(tlm_inputs["attention_mask"]))
            #print("TLM PREFILL INPUT IDs: ",tlm_inputs["input_ids"])
            #print("TLM PREFILL POSITION IDs",tlm_inputs["position_ids"])
            #print("TLM PREFILL CACHE INDEX",tlm_inputs["cache_index"])
            
            #run prefill on TLM and get the logits and last token from here (used to evaluate the first speculated token)
            tlm_session.set_buffers({"logits":np.zeros((batch_size, num_tokens, vocab_size), dtype=np.float16)})
            
            tlm_devrun_start = perf_counter()
            #bp()
            tlm_prefill_outputs = tlm_session.run(tlm_inputs)
            tlm_devrun_end = perf_counter()
            total_tlm_devrun_time+=tlm_devrun_end-tlm_devrun_start
            total_tlm_iter+=1

            logits_out_tlm_prefill = tlm_prefill_outputs["logits"]
            if len(logits_out_tlm_prefill.shape) == 2:
                logits_out_tlm_prefill = np.expand_dims(logits_out_tlm_prefill, 1)
            #print("TLM PREFILL LOGITS SHAPE ", logits_out_tlm_prefill.shape)
            #print("TLM PREFILL OUTPUTS ARGMAX ", logits_out_tlm_prefill.argmax(2))
            # print("TLM PREFILL ALL 32 TOKENS ",tokenizer.batch_decode(logits_out_tlm_prefill.argmax(2)))
            #first token speculated will be evaluated using the logits output at the last token position of the prefill
            tlm_prefill_logit_out = logits_out_tlm_prefill[:,-1:]
            #print("AFTER PREFILL NEXT TOKEN ID",next_token_id)
            tlm_cache_index += num_tokens

            tlm_prefill_etime = perf_counter()
            #print(f"***Timer TLM Prefill : {1 / (tlm_prefill_etime - tlm_prefill_stime)}")

        #PRECODE tlm evaluation + new_token
        # check if this is the first precode pass
        #print("*** TLM Precode ***")
        tlm_precode_stime = perf_counter()
        tlm_inputs["cache_index"] = tlm_cache_index
        tlm_position_id_start = None
        if tlm_cache_index[0] == prompt_len and is_first_iter:
            is_first_iter=False
            #just performed a PREFILL earlier in this iteration
            #get position ids by summing the attention mask
            num_tlm_input_tokens = max_spec_length #in the case that we just performed a prefill, we will provide only the current iter speculation as the input tokens
            tlm_position_id_start = tlm_inputs["attention_mask"].sum(1, keepdims=True)
            tlm_inputs["input_ids"] = dlm_candidate_ids
        else:
            #prefill was not done in this spd iter, so we need to provide the prev iter's TLM-token as input to get it's KV$
            #print("OVERALL NUM TOKENS ACCEPTED = ", overall_num_tokens_accepted)
            #TODO fix for batch_size>1
            tlm_position_id_start = prompt_num_valid_tokens + overall_num_tokens_accepted - 1
            tlm_inputs["input_ids"] = np.concatenate((tlm_last_iter_newtoken_id, dlm_candidate_ids),axis=1)

        #TODO fix for batch_size > 1
        tlm_inputs["position_ids"] = np.expand_dims(np.arange(tlm_position_id_start,tlm_position_id_start+num_tlm_input_tokens),axis=0)
        #print("tlm_pos_id_start",tlm_position_id_start)
        #print("tlm_position_id_end",tlm_position_id_start+num_tlm_input_tokens)
        #print(tlm_inputs["position_ids"])
        # if(num_tlm_input_tokens < (max_spec_length)):
        #     num_tlm_input_tokens = max_spec_length
        #     position_id_padding = np.zeros((batch_size, (max_spec_length) - tlm_inputs["position_ids"].shape[1]))
        #     input_id_padding = tokenizer.pad_token_id * np.ones((batch_size, (max_spec_length) - tlm_inputs["position_ids"].shape[1]))
        #     tlm_inputs["position_ids"] = np.concatenate((tlm_inputs["position_ids"], position_id_padding), axis=1)
        #     tlm_inputs["input_ids"] = np.concatenate((tlm_inputs["input_ids"],input_id_padding), axis = 1)
        # update the attention mask with valid positions set to 1 : prefill + tokens generated till now + speculated positions
        tlm_inputs["attention_mask"][0,prompt_len:prompt_len+overall_num_tokens_accepted+num_tlm_input_tokens-1] = 1
        #print("TLM PRECODE INPUT IDs: ",tlm_inputs["input_ids"])
        #print("TLM PRECODE POSITION IDs",tlm_inputs["position_ids"])
        #print("TLM PRECODE CACHE INDEX",tlm_inputs["cache_index"])
        #print("TLM PRECODE INPUT ATTENTION MASK ", tlm_inputs["attention_mask"].astype(int))
        #print("TLM PRECODE ATTENTION MASK length: ",np.sum(tlm_inputs["attention_mask"]))
        # EXPERIMENT WITH SPECIALIZATION OF SPEC LENGTH
        tlm_session.set_buffers({"logits":np.zeros((batch_size, num_tlm_input_tokens, vocab_size), dtype=np.float16)})
        update_time("tlm_preproc",tlm_precode_stime)
 
        tlm_devrun_start = perf_counter()
        tlm_precode_outputs = tlm_session.run(tlm_inputs)
        update_time("tlm_run", tlm_devrun_start)
        total_tlm_iter+=1
        #Can modify this if TLM outputs perfect shapes TBD:
        logits_out_tlm_precode = tlm_precode_outputs["logits"][:,:num_tlm_input_tokens]#does this do it?
        #print("TLM PRECODE OUTPUT LOGITS SHAPE ", logits_out_tlm_precode.shape)
        #print(logits_out_tlm_precode.argmax(2))
        # print(tokenizer.batch_decode(logits_out_tlm_precode.argmax(2)))
        tlm_cache_index += num_tlm_input_tokens
        # retrieve logits for the first spec_length + 1 positions (since we pad to the right here)
        if is_tlm_prefill:
            #the first spec_length-1 logits of this current PRECODE are the q(x2) till q(x_spec_length)
            # q(x1) was obtained as the output of the PREFILL stage that happened earlier in this same iteration
            # the last logit is for the new_token_if_all_prev_accepted
            #print("Shape of logits from precode", logits_out_tlm_precode.shape)
            tlm_logits = np.concatenate((tlm_prefill_logit_out,logits_out_tlm_precode[:,:spec_length]), axis=1)
            #print(tlm_logits[:,0:1].argmax(2))
            #print("TLM LOGITS AFTER PREFILL + PRECODE",tlm_logits.argmax(2))
        else:
            # the first spec_length logits are the q(x1) till q(x_spec_length)
            # the last token is the q(new_token_if_all_prev_accepted)
            tlm_logits = logits_out_tlm_precode[:,:spec_length+1]
        # last position is to be used for sampling the new token
        # the previous spec_length to be used for MRS on the speculated logits
        #print(f"***Timer Inference TLM Precode : {1 / (tlm_precode_etime_infer - tlm_precode_stime_infer)}")
        #print(f"***Timer TLM Precode : {1 / (tlm_precode_etime - tlm_precode_stime)}")
        #print(tlm_logits.shape)
        # print(tokenizer.batch_decode(tlm_logits.argmax(2)))

        #print("\n**** Candidate Selection ****")
        candidate_sel_stime = perf_counter()
        #TODO fix for batch_size > 1
        select_upto = 0#np.random.randint(1, spec_length+1)
        for spec_idx in range(spec_length):
            debugprint(f"MRS loop at speculation index : {spec_idx}")
            correct_tlm_sample_token = tokenizer.pad_token_id
            draft_token = np.expand_dims(dlm_candidate_ids[:,spec_idx],1)
            is_accepted=False
            if greedy:
                greedy_tlm_stime=perf_counter()
                tlm_token = np.argmax(np.expand_dims(tlm_logits[:,spec_idx],1), axis=-1)
                update_time("tlm_sample", greedy_tlm_stime)
                correct_tlm_sample_token = tlm_token
                # print("tlm token sampled: ",tlm_token)
                # print("corresponding draft token: ",draft_token)
                exact_match_stime=perf_counter()
                is_accepted = tlm_token == draft_token
                update_time("exact_match", exact_match_stime)
            else:
                tlm_logit_warp_start=perf_counter()
                if target_to_sample:
                    logits_out_tlm_warped = get_warped_logits(logits_np=tlm_logits[:,spec_idx:spec_idx+1], top_k=target_top_k, top_p=target_top_p, topk_logit_warper=target_topk_logit_warper, topp_logit_warper=target_topp_logit_warper, temperature=temperature_target, max_num_tokens=max_warped_tokens)
                else:
                    logits_out_tlm_warped = tlm_logits[:,spec_idx:spec_idx+1].astype(np.float32)
                tlm_warped_logits[:,spec_idx:spec_idx+1] = logits_out_tlm_warped[0]

                update_time("tlm_logit_warp", tlm_logit_warp_start)
                debugprint("Sampling Target token")
                tlm_sample_stime=perf_counter()
                tlm_token = sample_from_logits(logits_np=logits_out_tlm_warped, do_sample=target_to_sample, use_gumbel=to_use_gumbel)
                #log_softmax takes in a pytorch tensor and a dimension.  then at the end we want to convert that back to numpy.
                update_time("tlm_sample",tlm_sample_stime)
                log_smax_stime=perf_counter()
                logp_draft  = torch.log_softmax(torch.tensor(np.array(dlm_warped_logits[:,spec_idx], dtype=np.float32)), dim=-1).numpy()
                logp_target = torch.log_softmax(torch.tensor(np.array(tlm_warped_logits[:,spec_idx], dtype=np.float32)), dim=-1).numpy()
                update_time("log_softmax", log_smax_stime)
                #tlm_token = sample_from_logits(tlm_logits[:,spec_idx:spec_idx+1], target_to_sample, to_use_gumbel)
                #tlm_greedy_token_id = sample_from_logits(tlm_logits[:,spec_idx:spec_idx+1],0, to_use_gumbel)
                #print("At %d totally and spec_idx %d TLM next_token_id: %.1d" % (overall_num_tokens_accepted, spec_idx, tlm_token), file = debug_file)
                #print(tokenizer.batch_decode(tlm_token), file = debug_file);
                #print("Getting draft logprob")
                #dlm_warped_logits[:,spec_idx]  <-- numpy.
                #torch.tensor(np.array(dlm_warped_logits[:,spec_idx], dtype=np.float32))
                mrs_stime=perf_counter()
                is_accepted, correct_tlm_sample_token = rejection_sampling(token=draft_token, logp_draft_np=logp_draft, logp_target_np=logp_target, do_sample_draft=draft_to_sample, do_sample_main=target_to_sample, use_gumbel=to_use_gumbel, target_token=tlm_token, sample_matching=is_exact_sample_matching)
                update_time("mrs", mrs_stime)
            if is_accepted:
                #print(f"TLM has accepted token id {draft_token} at index {spec_idx} from Draft")
                select_upto += 1
            else:
                #print(f"TLM has rejected token id {draft_token} at index {spec_idx} from Draft")
                new_tlm_id = correct_tlm_sample_token
                #print("New TLM Id = ", new_tlm_id)
                break
        if DEBUG == 1:
            if select_upto not in n_tokens_accepted_histogram.keys():
                n_tokens_accepted_histogram[select_upto] = 0
            n_tokens_accepted_histogram[select_upto] += 1
        debugprint("Number of tokens accepted this MRS round:", select_upto)
        debugprint("prompt_len+overall_num_tokens_accepted=",(prompt_len+overall_num_tokens_accepted))
        # print(f"Selected the first {select_upto} speculated tokens from DLM")
        # add selected candidate ids to the generated ids
        #TODO fix for batch_size > 1
        output_update_stime=perf_counter()
        if select_upto > 0:
            generated_ids[0,overall_num_tokens_accepted:overall_num_tokens_accepted+select_upto] = dlm_candidate_ids[0,:select_upto]
            if n_ttft == 0:
                first_token_time = perf_counter()
                ttft  = first_token_time - start
                # print("TTFT = %.2f" % ttft)
                #print("TTFT = %.2f" % ttft, file = f)
            n_ttft = n_ttft + 1
        #if all speculated tokens were accepted, append the new TLM-generated id here
        if select_upto == spec_length:
            if dlm_candidate_ids[0,spec_length-1] == tokenizer.eos_token_id or spec_length<max_spec_length:
                #stop here, dont sample new token from TLM since we have either generated an eos or are at the end of ctx_len
                pass
            elif greedy:
                new_tlm_id = np.argmax(tlm_logits[:,select_upto:select_upto+1],axis=-1)
            else:
                if target_to_sample:
                    logits_out_tlm_warped = get_warped_logits(logits_np=tlm_logits[:,select_upto:select_upto+1], top_k=target_top_k, top_p=target_top_p, topk_logit_warper=target_topk_logit_warper, topp_logit_warper=target_topp_logit_warper, temperature=temperature_target, max_num_tokens=max_warped_tokens)
                else:
                    logits_out_tlm_warped = tlm_logits[:,select_upto:select_upto+1].astype(np.float32)
                #agokhale fix the sampling of the extra token by warping the tlm logits here as well
                new_tlm_id = sample_from_logits(logits_np=logits_out_tlm_warped, do_sample=target_to_sample, use_gumbel=to_use_gumbel)
        this_iter_gen_ids = list(generated_ids[0, overall_num_tokens_accepted:overall_num_tokens_accepted+select_upto]) + list(new_tlm_id[0])

        # for this_iterid in this_iter_gen_ids:
        #     print(f"{this_iterid}:{tokenizer.decode(this_iterid)}|")
        if stream:
            #print("*"*100)
            color = "white"
            if model_family == "llama2":
                nextlinecount = 0
                #use convert ids to tokens here to find and utilize the prefix-spaces to decode correct sentences
                curr_token_list = tokenizer.convert_ids_to_tokens(this_iter_gen_ids, skip_special_tokens=True)
                for i,tok in enumerate(curr_token_list):
                    if i == (len(curr_token_list)-1):
                        color="white"
                    else:
                        color="deep_sky_blue1"
                    curr_token = tok
                    curr_token = curr_token.replace("<0x0A>","\n")
                    if curr_token == "\n":
                        nextlinecount += 1
                        if nextlinecount > 3:
                            continue
                    else:
                        nextlinecount = 0
                    if ord(curr_token[0]) == UNDERSCORE_SPECIAL_TOKEN:
                        console.print(f"[{color}] {curr_token[1:]}",end="", highlight=False)

                    else:
                        console.print(f"[{color}]{curr_token}",end="", highlight=False)

            else:
                for thisiterid in this_iter_gen_ids:
                    console.print(f"[{color}]{tokenizer.decode(thisiterid)}", end="")


        overall_num_tokens_accepted += select_upto
        if new_tlm_id is not None:
            #print(overall_num_tokens_accepted)
            #print(select_upto)
            # print("NEW TLM ID")
            # print(tokenizer.batch_decode(new_tlm_id))
            generated_ids[0,overall_num_tokens_accepted] = new_tlm_id
            next_token_id = new_tlm_id
            tlm_last_iter_newtoken_id = new_tlm_id
            overall_num_tokens_accepted += 1

        num_dlm_tokens_rejected = spec_length - select_upto
        #print("DLM cache index originally ",dlm_cache_index)


        #TODO append the new token from TLM into the list of generated_ids
        # name it new_tlm_id and remove the assignment to it from the else block below
        # also add copy to tlm_last_accepted_id

        # update input ids, position ids and cache indices for DLM and TLM for the next pass
        #print("*** Updating inputs, position ids and cache index based on MRS ***")
        if num_dlm_tokens_rejected > 0:
            #DLM cache rewind
            assert(dlm_inputs["position_ids"].shape[1] == 1)
            dlm_cache_index -= (num_dlm_tokens_rejected - 1)
            dlm_inputs["position_ids"] -= (num_dlm_tokens_rejected - 1) #since the input in next iter is the new tlm token from this iter
            #print("DLM cache index updated to ",dlm_cache_index)
            #print("Not all tokens accepted, rewind the kv cache pointer and position ids")
            ##TLM cache rewind
            #print("TLM CACHE INDEX ORIGINALLY**", tlm_cache_index)
            tlm_cache_index -= (num_dlm_tokens_rejected)
            #print("TLM CACHE INDEX UPDATED**", tlm_cache_index)

        else:
            #DLM precode to be run: if all tokens were accepted, then we need to generate KV$ for the last speculated token
            #print("All tokens accepted, passing seq_len = 2 to DLM for the first fwdprop of the next iter")
            # TODO fix for batch_size > 1
            dlm_inputs["input_ids"] = np.concatenate((dlm_candidate_ids[:,-1:],new_tlm_id), axis=1)
            dlm_inputs["position_ids"] = np.concatenate((dlm_inputs["position_ids"]+1, dlm_inputs["position_ids"]+2), axis=1)
            
        candidate_sel_etime = perf_counter()
        update_time("output_cache_update", output_update_stime)
        #print(f"***Timer Candidate Selection : {1 / (candidate_sel_etime - candidate_sel_stime)}")
        #print(f"**** SPS ITER {num_iters} DONE ****")
        # print(tokenizer.batch_decode(generated_ids))
        num_iters+=1
        tlm_infer_end = perf_counter()
        curr_tlm_time = tlm_infer_end - tlm_infer_start
        total_tlm_time = total_tlm_time + curr_tlm_time 
        is_tlm_prefill = False
        if (new_tlm_id == tokenizer.eos_token_id).all():
            stop_generation = True
        if len(stop_sequences) > 0:
            output_text = tokenizer.batch_decode(generated_ids)

            for stop_sequence in stop_sequences:
                if stop_sequence in output_text[0]:
                    stop_generation = True

        if stop_generation:
            break
    #print(overall_num_tokens_accepted / (perf_counter() - time1))
    #print("Exited the Spec Decode Loop:")
    end = perf_counter()
    # print("Average Tokens Generated per TLM inference:", overall_num_tokens_accepted / num_iters)
    #print("Average Tokens Generated per TLM inference:", overall_num_tokens_accepted / num_iters, file = f)
    total_perf = end - start
    #print("Total time taken: %.5f " % total_perf, file = f)
    #print("TLM stats---------------------------------", file = f)
    #print("Total tlm iter: %.5d " % total_tlm_iter, file = f)
    #print("Total TLM time: %.5f " % total_tlm_time, file = f)
    #print("Total TLM device only time: %.5f " % total_tlm_devrun_time, file=f)
    tlm_overhead = total_tlm_time - total_tlm_devrun_time
    #print("Total TLM overhead time: %.5f " % tlm_overhead, file=f)
    tlm_time_per_iter = total_tlm_time / total_tlm_iter
    #print("Total time per tlm iter: %.5f " % tlm_time_per_iter, file = f)
    tlm_iter_per_time = 1 / tlm_time_per_iter
    #print("Total tlm iter per sec: %.5f " % tlm_iter_per_time, file = f)
    tlm_time_per_iter = total_tlm_devrun_time / total_tlm_iter
    #print("Total devrun time per tlm iter: %.5f " % tlm_time_per_iter, file = f)
    tlm_iter_per_time = 1 / tlm_time_per_iter
    #print("Total devrun tlm iter per sec: %.5f " % tlm_iter_per_time, file = f)

    #print("DLM stats---------------------------------", file = f)
    #print("Total dlm iter: %.5d " % total_dlm_iter, file = f)
    #print("Total DLM time: %.5f " % total_dlm_time, file = f)
    #print("Total DLM device only time: %.5f " % total_dlm_devrun_time, file=f)
    dlm_overhead = total_dlm_time - total_dlm_devrun_time
    #print("Total DLM overhead time: %.5f " % dlm_overhead, file=f)
    dlm_time_per_iter = total_dlm_time / total_dlm_iter
    #print("Total time per dlm iter: %.5f " % dlm_time_per_iter, file = f)
    dlm_iter_per_time = 1 / dlm_time_per_iter
    #print("Total dlm iter per sec: %.5f " % dlm_iter_per_time, file = f)
    dlm_time_per_iter = total_dlm_devrun_time / total_dlm_iter
    #print("Total devrun time per dlm iter: %.5f " % dlm_time_per_iter, file = f)
    dlm_iter_per_time = 1 / dlm_time_per_iter
    #print("Total devrun dlm iter per sec: %.5f " % dlm_iter_per_time, file = f)


    # print("overall num of tokens accepted %.2d" % overall_num_tokens_accepted)
    # print("overall time spent %.5f" % (end - start))
    total_perf = (overall_num_tokens_accepted) / (end - start)
    decode_only_perf = (overall_num_tokens_accepted - tokens_accepted_in_first_round) / (end - start_decode_throughput_counter)
    # generated_texts = tokenizer.batch_decode(generated_ids)
    # debugprint(generated_texts[0])
    debugprint()
    print()
    print("="*60)
    print("Token Generation Rate:\t", round(decode_only_perf, 2), "tokens per second")
    print("Acceptance Rate:\t", round(overall_num_tokens_accepted/total_tlm_iter, 2))
    print("="*60)
    print()
    print()
    if DEBUG == 1:
        print(time_dict)
        print(n_tokens_accepted_histogram)

if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument("--model-family", required=True, help="Model family, choose from \"llama2\" or \"codegen\"")
    argp.add_argument("--model-name", required=True, help="Model name to run")
    argp.add_argument("--prompt-len", type=int, default=32, help="Prompt length")
    argp.add_argument("--ctx-len", type=int, default=128, help="Context length")
    argp.add_argument("--dlm-qpc", required=True, help="Compiled binary DLM QPC")
    argp.add_argument("--tlm-qpc", required=True, help="Compiled binary TLM QPC")
    argp.add_argument(
        "--prompt",
        default="My name is",
        help="Input prompt(s) to generate for (pipe-separated)",
    )
    argp.add_argument(
        "--no-stream", action="store_false", dest="stream", help="Don't stream output text"
    )
    argp.add_argument(
        "--device_id",
        default=[0],
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        help="QAIC device ids (comma-separated)",
    )
    #argp.add_argument("--temperature", default=1.0, help="Draft sampling temperature for log prob")
    argp.add_argument("--write-io-dir", help="Directory to write inputs/outputs into")
    argp.add_argument("--exact-greedy", action="store_true", dest="greedy", help="Pass this to run exact matching on greedily-sampled tokens from TLM and DLM")
    argp.add_argument("--max-spec-length", type=int, default=7, dest="max_spec_length", help="Length of the speculation generated by the draft LM")
    args = argp.parse_args()
    main(**vars(args))
