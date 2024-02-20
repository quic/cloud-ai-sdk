# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import json
import os
from time import perf_counter
from typing import Dict, List, Optional
import numpy as np
import transformers
import sys
from qaic_infer import QAICInferenceSession

io_files = []
def write_io_files(
    inputs: Dict[str, np.ndarray],
    outputs: Dict[str, np.ndarray],
    write_io_dir: str,
    write_io_subdir: str,
    write_io_name: str,
    include_dims: bool = False,
    reset: bool = False,
):
    global io_files
    if reset:
        io_files = []
    io = []
    os.makedirs(f"{write_io_dir}/{write_io_subdir}", exist_ok=True)
    for iname, iarray in inputs.items():
        iarray.tofile(f"{write_io_dir}/{write_io_subdir}/{iname}.raw")
        ispec = {
            "path": f"{write_io_subdir}/{iname}.raw",
            "io-direction": "in",
            "elem-size": iarray.itemsize,
            "map-to": iname,
        }
        if include_dims:
            ispec["dims"] = iarray.shape
        io.append(ispec)
    for oname, oarray in outputs.items():
        oarray.tofile(f"{write_io_dir}/{write_io_subdir}/{oname}.raw")
        ospec = {
            "path": f"{write_io_subdir}/{oname}.raw",
            "io-direction": "out",
            "elem-size": oarray.itemsize,
            "map-to": oname,
        }
        if include_dims or oname.endswith("_RetainedState"):
            ospec["dims"] = oarray.shape
        io.append(ospec)
    io_files.append(io)
    with open(f"{write_io_dir}/{write_io_name}.json", "w") as fp:
        json.dump({"IO-files": io_files}, fp, indent=True)
def main(
    model_name: str,
    qpc: str,
    prompt: List[str],
    stream: bool = True,
    device_id: List[int] = [0],
    enable_debug_logs: bool = False,
    write_io_dir: Optional[str] = None,
    automation: bool = False,
) -> Dict[str, float]:
    # Load QPC
    session = QAICInferenceSession(qpc, device_id, enable_debug_logs=enable_debug_logs)
    #session = QAICInferenceSession(qpc, device_id, enable_debug=enable_debug_logs)
    # Read prompt and ctx len from session
    if  len(session.allowed_shapes)>0:
        prompt_len = max(
            [x[session.binding_index_map["input_ids"]][1][1] for x in session.allowed_shapes]
        )
        ctx_len = session.allowed_shapes[0][session.binding_index_map["attention_mask"]][1][1]
    else:
        prompt_len = 1
        ctx_len = session.bindings[session.binding_index_map["attention_mask"]].dims[1]
    # Skip inputs/outputs
    session.skip_buffers([x for x in session.input_names if x.startswith("past_")])
    session.skip_buffers([x for x in session.output_names if x.endswith("_RetainedState")])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tok_prompt = [tokenizer.encode(x) for x in prompt]
    max_prompt_len = len(max(tok_prompt,key=len))
    if max_prompt_len > 1:
        num_chunks = -(max_prompt_len// -prompt_len)
        input_len= num_chunks * prompt_len
    else:
        num_chunks = 1
        input_len= prompt_len


    print("==============================Start of Inference=================================")
    print("Input Prompt length = "+ str(input_len))
    print("---------------------------------------------------------------------------")
    # Prepare inputs for first iteration
    start = perf_counter()
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=input_len)
    batch_size = inputs["input_ids"].shape[0]
    inputs["position_ids"] = (np.cumsum(inputs["attention_mask"], 1) - 1) * inputs["attention_mask"]
    inputs["attention_mask"] = np.concatenate(
        [
            inputs["attention_mask"].astype(bool),
            np.zeros((batch_size, ctx_len - input_len), dtype=bool),
        ],
        1,
    )
    cache_index = np.array([0])
    inputs["cache_index"] = cache_index
    generated_ids = np.full((batch_size, ctx_len - input_len + 1), tokenizer.pad_token_id)
    if stream:
        print(0, prompt[0], end="", flush=True)
    # Run prefill
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prompt_len : (i + 1) * prompt_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][
            :, i * prompt_len : (i + 1) * prompt_len
        ]
        chunk_inputs["attention_mask"] = inputs["attention_mask"].copy()
        chunk_inputs["attention_mask"][:, (i + 1) * prompt_len :] = False
        outputs = session.run(chunk_inputs)
        if write_io_dir:
            write_io_files(inputs, outputs, write_io_dir, "prefill", "aic_batch_io", True, False)
        cache_index += prompt_len
    # Get first token
    logits = outputs["logits"]
    if len(logits.shape) == 2:
        logits = np.expand_dims(logits, 1)
    next_token_id = logits.argmax(2)
    inputs["input_ids"] = next_token_id
    inputs["position_ids"] = inputs.pop("attention_mask").sum(1, keepdims=True)
    generated_ids[:, cache_index[0] - input_len] = next_token_id.squeeze(1)
    if stream:
        print(tokenizer.decode(next_token_id[0]), end="", flush=True)
    # Skip attention_mask from next iteration to use retained attention_mask
    session.skip_buffers(["attention_mask"])
    loop_start = perf_counter()
    while (next_token_id != tokenizer.eos_token_id).all() and cache_index[0] < ctx_len:
        outputs = session.run(inputs)
        if write_io_dir:
            write_io_files(inputs, outputs, write_io_dir, "decode", "aic_batch_io", True, False)
            write_io_dir = None
        # Prepare inputs for next iteration
        logits = outputs["logits"]
        if len(logits.shape) == 2:
            logits = np.expand_dims(logits, 1)
        next_token_id = logits.argmax(2)
        inputs["input_ids"] = next_token_id
        inputs["position_ids"] += 1
        cache_index += 1
        generated_ids[:, cache_index[0] - input_len] = next_token_id.squeeze(1)
        if stream:
            print(tokenizer.decode(next_token_id[0]), end="", flush=True)
    end = perf_counter()
    generated_texts = tokenizer.batch_decode(generated_ids)
    for i in range(1 if stream else 0, batch_size):
        print()
        print(i, prompt[i], generated_texts[i])
    prefill_perf = loop_start - start
    decode_perf = (cache_index.item() - input_len - 1) / (end - loop_start)
    total_perf = (cache_index.item() - input_len) / (end - start)
    print()
    print("------------Performance---------------")
    if automation:
        print("output=", generated_texts)
        print("Output Token/sec per Card =", round(decode_perf * batch_size, 2))
        return
    print("Output Tokens/sec per Card =", round(decode_perf * batch_size, 2))
    print("")
    print("Note:An AI 100 Cloud instance/server typically consists of 2/4/8/16 or more accelerator cards. The Output Tokens/sec should be scaled linearly to get the Output Tokens/sec per instance/server. For example, the Output tokens/sec should be multiplied by 8 for an AWS DL2q instance.")
    print("=========================END==============================")

if __name__ == "__main__":
    import argparse
    argp = argparse.ArgumentParser()
    argp.add_argument("--model-name", required=True, help="Model name to run")
    argp.add_argument("--qpc", required=True, help="Compiled binary QPC")
    argp.add_argument(
        "--prompt",
        type=lambda prompt: prompt.split("|"),
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
    argp.add_argument("--enable-debug-logs", action="store_true", help="Enable debug logs in LRT")
    argp.add_argument("--write-io-dir", help="Directory to write inputs/outputs into")
    argp.add_argument(
        "--automation", action="store_true", help="Print outputs in required format for automation"
    )
    args = argp.parse_args()
    main(**vars(args))
