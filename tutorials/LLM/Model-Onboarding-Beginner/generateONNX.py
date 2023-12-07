# -----------------------------------------------------------------------------
#
# Qualcomm Technologies, Inc. Proprietary
# (c) 2023 Qualcomm Technologies, Inc. All rights reserved.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
# -----------------------------------------------------------------------------

import os
from typing import Optional, Dict, List
import numpy as np
import torch
import transformers

from generateModel import (
    arg_parser,
    fix_onnx_fp16,
    generate_input_files,
    run_model_on_aic,
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
            dynamic_axes[iname] = {0: "batch_size", 2: "ctx_len"}
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
    model_path: Optional[str] = None,
    use_auth_token: bool = False,
    run_on_aic: bool = False,
):

    # Determine model save path
    model_base_name = model_name.split("/")[-1]
    model_base_name += "-kv"
    if model_path is None:
        model_path = model_base_name

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = model_class.from_pretrained(model_name, use_cache=True, use_auth_token=use_auth_token)
    model.eval()

    # Preprocess inputs
    input_str = ["My name is Sarah.", "I live in London."]
    if seq_len > 0:
        inputs = tokenizer(input_str, return_tensors="pt", padding=True)
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["input_ids"] = torch.concat(
            [
                inputs["input_ids"],
                torch.full((batch_size, seq_len - prompt_len), tokenizer.pad_token_id),
            ],
            1,
        )
        inputs["attention_mask"] = torch.concat(
            [
                inputs["attention_mask"],
                torch.zeros((batch_size, seq_len - prompt_len), dtype=torch.int64),
            ],
            1,
        )
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(1) - 1) * inputs["attention_mask"]
    else:
        inputs = tokenizer(input_str, return_tensors="pt")

    # Run PyTorch inference for first iteration
    pt_outputs = model(**inputs)
    output_names = list(pt_outputs.keys())

    # Raise error if expected outputs are not present
    assert "logits" in output_names, "logits not found in output"
    assert "past_key_values" in output_names, "past_key_values not found in output"

    # Build inputs for next iteration from outputs
    cache_index = torch.tensor(prompt_len)
    # inputs["input_ids"] = pt_outputs.logits.detach().argmax(2)
    inputs["input_ids"] = tokenizer(["I have"] * 2, return_tensors="pt").input_ids[:, -2:]
    inputs["position_ids"] = inputs["attention_mask"].sum(1, keepdim=True)
    inputs["position_ids"] = inputs["position_ids"].repeat(1, 2) + torch.arange(2).view(1, 2)
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
    fp32_model_name = simplify_onnx(gen_models_path, fp32_model_name, mutable_initializer=True)

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

    if not run_on_aic:
        return

    # onnx dynamic shape
    batch_size, seq_len = inputs["input_ids"].shape
    onnx_symbol_defs = {"batch_size": batch_size, "seq_len": seq_len}
    ctx_len = inputs["past_key.0"].shape[2]
    onnx_symbol_defs["ctx_len"] = ctx_len

    # AICOutputs
    write_output_dir = f"{model_path}/AICOutputs"
    os.makedirs(f"{write_output_dir}/FP16", exist_ok=True)

    # Run on AIC in FP16
    assert run_model_on_aic(
        f"{gen_models_path}/{fp16_model_name}.onnx",
        onnx_symbol_defs=onnx_symbol_defs,
        input_list_file=input_list_file,
        convert_to_fp16=True,
        write_output_dir=f"{write_output_dir}/FP16",
    ), "Compilation failed"

    # Verify outputs
    print("ONNXRT vs. AIC (MAD)")
    for oname, orto in zip(output_names, ort_outputs):
        aico16 = np.fromfile(
            f"{write_output_dir}/FP16/{oname}-activation-0-inf-0.bin", orto.dtype
        ).reshape(orto.shape)
        diff16 = np.abs(orto.astype(np.float32) - aico16.astype(np.float32)).max()
        print(oname, "FP16:", diff16)


if __name__ == "__main__":
    argp = arg_parser()
    args = argp.parse_args()
    main(**vars(args))
