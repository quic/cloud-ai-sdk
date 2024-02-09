# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from typing import Optional, Dict, List
import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download

from generateModel import (
    arg_parser,
    fix_onnx_fp16,
    generate_input_files,
    run_model_on_ort,
    simplify_onnx,
)


def export_onnx(
    pt_model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    gen_models_path: str,
    model_base_name: str,
) -> str:
    # Inspect the model's forward method arguments
    pt_model_code = pt_model.forward.__code__
    pt_input_names = pt_model_code.co_varnames[1 : pt_model_code.co_argcount]

    # Arrange the inputs in proper order to make tracing work properly
    pt_inputs = []
    input_names = []
    for input_name in pt_input_names:
        if input_name in inputs:
            if input_name == "past_key_values":
                for i in range(len(inputs[input_name])):
                    input_names.append(f"past_key.{i}")
                    input_names.append(f"past_value.{i}")
            else:
                input_names.append(input_name)
            pt_inputs.append(inputs[input_name])
        else:
            pt_inputs.append(None)

    # Create dynamic axes dict for inputs that need to have dynamic input shapes
    seq_len_inputs = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "encoder_outputs",
    }
    decoder_seq_inputs = {"decoder_input_ids", "decoder_attention_mask"}

    dynamic_axes = {}
    for iname in input_names:
        if iname in seq_len_inputs:
            dynamic_axes[iname] = {0: "batch_size", 1: "seq_len"}
        elif iname in decoder_seq_inputs:
            dynamic_axes[iname] = {0: "batch_size", 1: "decoder_seq_len"}
        elif iname.startswith("past_"):
            # KV-cache (batch_size, num_heads, past_len, embed_dim)
            dynamic_axes[iname] = {0: "batch_size", 1: "ctx_len"}
    if "past_key.0" in input_names and "attention_mask" in input_names:
        dynamic_axes["attention_mask"] = {0: "batch_size", 1: "ctx_len"}

    torch.onnx.export(
        pt_model,
        tuple(pt_inputs),
        f"{gen_models_path}/{model_base_name}.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    return model_base_name


def main(
    model_name: str,
    model_class: type,
    input_str: str,
    seq_len: int = 128,
    auth_token: Optional[str] = None,
    model_path: Optional[str] = None,
    run_on_aic: bool = False,
):

    # Determine model save path
    model_base_name = model_name.split("/")[-1]
    model_base_name += "-kv"
    if model_path is None:
        model_path = model_base_name

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=auth_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    snapshot_download(repo_id=model_name, token=auth_token)
    model = model_class.from_pretrained(model_name, use_cache=True, token=auth_token)
    model.eval()

    # Preprocess inputs
    assert seq_len > 0, "Need seq_len to be greater than zero"
    inputs = tokenizer([input_str] * 2, return_tensors="pt", padding="max_length", max_length=seq_len)
    batch_size = inputs["input_ids"].shape[0]
    prompt_len = inputs["attention_mask"].sum(1).max().item()
    inputs["position_ids"] = (inputs["attention_mask"].cumsum(1) - 1) * inputs["attention_mask"]
    inputs["attention_mask"] = inputs["attention_mask"].bool()

    # Run PyTorch inference for first iteration
    pt_outputs = model(**inputs)
    output_names = list(pt_outputs.keys())

    # Raise error if expected outputs are not present
    assert "logits" in output_names, "logits not found in output"
    assert "past_key_values" in output_names, "past_key_values not found in output"

    # Build inputs for next iteration from outputs
    cache_index = torch.tensor(prompt_len)
    inputs["input_ids"] = pt_outputs.logits.detach().argmax(2)
    inputs["input_ids"] = inputs["input_ids"].repeat(1, 2)
    inputs["position_ids"] = inputs["attention_mask"].sum(1, keepdim=True)
    inputs["position_ids"] = inputs["position_ids"].repeat(1, 2)
    inputs["attention_mask"] = inputs["attention_mask"].bool()
    inputs["cache_index"] = cache_index

    # Add past_key_values into inputs
    inputs["past_key_values"] = tuple(
        [(key.detach(), value.detach()) for key, value in pt_outputs.past_key_values]
    )

    # Run PyTorch inference with past
    pt_outputs = model(**inputs)
    output_names = list(pt_outputs.keys())

    # Add pkv into output_names
    pkv = tuple([(key.detach(), value.detach()) for key, value in pt_outputs.past_key_values])
    pkv_idx = output_names.index("past_key_values")
    key_value_names = [f"past_{x}.{i}" for i in range(len(pkv)) for x in ["key", "value"]]
    output_names[pkv_idx : pkv_idx + 1] = [x + "_RetainedState" for x in key_value_names]

    # Replace nested past_key_values outputs with separate KV tensors
    pt_outputs = dict(pt_outputs)
    pkv_out = pt_outputs.pop("past_key_values")
    for i, (key, value) in enumerate(pkv_out):
        pt_outputs[f"past_key.{i}_RetainedState"] = key
        pt_outputs[f"past_value.{i}_RetainedState"] = value

    # Export and simplify ONNX model
    gen_models_path = f"{model_path}/generatedModels"
    os.makedirs(gen_models_path, exist_ok=True)
    fp32_model_name = export_onnx(model, inputs, output_names, gen_models_path, model_base_name)
    # fp32_model_name = simplify_onnx(gen_models_path, fp32_model_name, mutable_initializer=True)

    # Replace nested past_key_values inputs with separate KV tensors
    inputs.pop("past_key_values")
    for i, (key, value) in enumerate(pkv):
        inputs[f"past_key.{i}"] = key
        inputs[f"past_value.{i}"] = value

    # Run ONNXRT inference
    input_names, ort_outputs = run_model_on_ort(
        f"{gen_models_path}/{fp32_model_name}.onnx", inputs, output_names
    )
    print("PyTorch vs. ONNXRT (MAD):")
    for oname, orto in zip(output_names, ort_outputs):
        pto = pt_outputs[oname].detach().numpy()
        print(oname, np.abs(pto.astype(np.float32) - orto.astype(np.float32)).max())

    # Fix onnx for FP16
    fp16_model_name = fix_onnx_fp16(
        inputs, output_names, ort_outputs, gen_models_path, fp32_model_name
    )

    # Generate custom-IO files
    with open(f"{model_path}/custom_io.yaml", "w") as fp:
        fp.write("# Model Inputs\n\n")
        for input_name in key_value_names:
            fp.write(f" - IOName: {input_name}\n   Precision: float16\n\n")
            inputs[input_name] = inputs[input_name].to(torch.float16)
        fp.write("# Model Outputs\n\n")
        for output_name in key_value_names:
            fp.write(f" - IOName: {output_name}_RetainedState\n   Precision: float16\n\n")

    # Generate inputFiles
    input_list_file = f"{model_path}/input_list.txt"
    generate_input_files(f"{model_path}/inputFiles", input_names, inputs, input_list_file)

    if run_on_aic:
        print("--run-on-aic not supported for this script")


if __name__ == "__main__":
    argp = arg_parser()
    args = argp.parse_args()
    main(**vars(args))
