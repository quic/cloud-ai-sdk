# -----------------------------------------------------------------------------
#
# Qualcomm Technologies, Inc. Proprietary
# (c) 2022 Qualcomm Technologies, Inc. All rights reserved.
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
import re
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime
import onnxsim
import torch
import transformers


def export_onnx(
    pt_model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    gen_models_path: str,
    model_base_name: str,
) -> str:
    pt_model_code = pt_model.forward.__code__
    pt_input_names = pt_model_code.co_varnames[1 : pt_model_code.co_argcount]

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
            dynamic_axes[iname] = {0: "batch_size", 1: "sequence"}
        elif iname in decoder_seq_inputs:
            dynamic_axes[iname] = {0: "batch_size", 1: "decoder_sequence"}
        elif iname.startswith("past_"):
            dynamic_axes[iname] = {0: "batch_size", 2: "past_sequence"}
    if "past_key.0" in input_names and "attention_mask" in input_names:
        dynamic_axes["attention_mask"] = {0: "batch_size", 1: "sequence+past_sequence"}

    torch.onnx.export(
        pt_model,
        tuple(pt_inputs),
        f"{gen_models_path}/{model_base_name}.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    return model_base_name


def save_onnx(model: onnx.ModelProto, gen_models_path: str, model_base_name: str):
    # Load the external tensors into the ModelProto, so the right size is calculated
    # and re-exported into right external tensor file
    onnx.load_external_data_for_model(model, gen_models_path)
    size_gb = model.ByteSize() / 1073741824

    if size_gb <= 2:
        onnx.save(model, f=f"{gen_models_path}/{model_base_name}.onnx")
    else:
        onnx.save(
            model,
            f=f"{gen_models_path}/{model_base_name}.onnx",
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{model_base_name}.onnx.data",
            convert_attribute=True,
        )


def simplify_onnx(gen_models_path: str, model_base_name: str, **kwargs) -> str:
    simple_model, check = onnxsim.simplify(f"{gen_models_path}/{model_base_name}.onnx", **kwargs)
    assert check, "Failed verification of simplified model"

    model_base_name += "_simplified"
    save_onnx(simple_model, gen_models_path, model_base_name)
    return model_base_name


def fix_onnx_fp16(
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    ort_outputs: List[np.ndarray],
    gen_models_path: str,
    model_base_name: str,
) -> str:
    finfo = np.finfo(np.float16)
    fp16_max = finfo.max
    fp16_min = finfo.min

    model = onnx.load(f"{gen_models_path}/{model_base_name}.onnx")
    fp16_fix = False
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        nptensor = numpy_helper.to_array(tensor, gen_models_path)
        if nptensor.dtype == np.float32 and (
            np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)
        ):
            nptensor = np.clip(nptensor, fp16_min, fp16_max)
            new_tensor = numpy_helper.from_array(nptensor, tensor.name)
            tensor.CopyFrom(new_tensor)
            fp16_fix = True

    if fp16_fix:
        # Save FP16 model
        print("Found constants out of FP16 range, clipped to FP16 range")
        model_base_name += "_fp16"
        save_onnx(model, gen_models_path, model_base_name)
        '''
        # Check if the FP16-fixed model can be used for FP32
        close_outputs = []
        _, ort_outputs_fixed = run_model_on_ort(
            f"{gen_models_path}/{model_base_name}.onnx", inputs, output_names
        )
        print("ONNXRT vs. ONNXRT fixed (MAD):")
        for oname, orto, ortof in zip(output_names, ort_outputs, ort_outputs_fixed):
            fix_diff = np.abs(orto - ortof).max()
            print(oname, fix_diff)
            close_outputs.append(fix_diff < 1e-5)

        # Replace FP32 model with FP16-fixed model
        if all(close_outputs):
            print("Using FP16 model for FP32")
            model_base_name = model_base_name[:-5]
            os.remove(f"{gen_models_path}/{model_base_name}.onnx")
            if os.path.exists(f"{gen_models_path}/{model_base_name}.onnx.data"):
                os.remove(f"{gen_models_path}/{model_base_name}.onnx.data")
            save_onnx(model, gen_models_path, model_base_name)
            os.remove(f"{gen_models_path}/{model_base_name}_fp16.onnx")
            if os.path.exists(f"{gen_models_path}/{model_base_name}_fp16.onnx.data"):
                os.remove(f"{gen_models_path}/{model_base_name}_fp16.onnx.data")

        '''
    return model_base_name


def generate_npi(
    original_precision_nodes_pattern: str,
    npi_path: str,
    gen_models_path: str,
    model_base_name: str,
):
    mixed_precision = False
    if original_precision_nodes_pattern:
        npi = open(npi_path, "w")
        npi.write("FP16NodeInstanceNames:\n")
        model = onnx.load(f"{gen_models_path}/{model_base_name}.onnx")
        fp16_nodes = 0
        for node in model.graph.node:
            if not re.search(original_precision_nodes_pattern, node.name):
                npi.write(f"  - {node.output[0]}\n")
                fp16_nodes += 1
        npi.close()
        if fp16_nodes == len(model.graph.node):
            print("Couldn't find nodes for FP32, run in FP16")
            os.remove(npi_path)
        else:
            mixed_precision = True
    return mixed_precision


def generate_input_files(
    input_files_path: str,
    input_names: List[str],
    inputs: Dict[str, torch.Tensor],
    input_list_file: str,
):
    # inputFiles
    os.makedirs(input_files_path, exist_ok=True)
    filenames = []
    for name in input_names:
        # We can't directly iterate with inputs.items() because
        # we have to maintain the order of input_names
        suffix = inputs[name].shape[-1] if len(inputs[name].shape) > 0 else 0
        filename = f"{input_files_path}/{name}_{suffix}.raw"
        inputs[name].detach().numpy().tofile(filename)
        filenames.append(filename.split("/", 1)[-1])

    # input_list.txt
    with open(input_list_file, "w") as fp:
        fp.write(",".join(filenames))
        fp.write("\n")


def run_model_on_ort(
    onnx_path: str,
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
) -> Tuple[List[str], List[np.ndarray]]:
    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_names = [x.name for x in ort_session.get_inputs()]
    ort_outputs = ort_session.run(
        output_names, {k: v.detach().numpy() for k, v in inputs.items() if k in input_names}
    )
    return input_names, ort_outputs


def run_model_on_aic(
    onnx_path: str,
    onnx_symbol_defs: Dict[str, int] = {},
    **kwargs,
) -> bool:
    args = [
        "/opt/qti-aic/exec/qaic-exec",
        f"-m={onnx_path}",
        "-aic-hw",
        "-aic-hw-version=2.0",
    ]
    for onnx_symbol, onnx_def in onnx_symbol_defs.items():
        args.append(f"-onnx-define-symbol={onnx_symbol},{onnx_def}")
    for k, v in kwargs.items():
        k = k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                args.append(f"-{k}")
            continue
        args.append(f"-{k}={v}")

    print("Running compiler:", " ".join(args))
    result = subprocess.run(args)
    return result.returncode == 0


def main(
    model_name: str,
    model_class: type,
    input_str: str,
    seq_len: int = 128,
    model_path: Optional[str] = None,
    original_precision_nodes_pattern: Optional[str] = None,
    run_on_aic: bool = False,
):

    # Determine model save path
    model_base_name = model_name.split("/")[-1]
    if model_path is None:
        model_path = model_base_name

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(input_str, model_class)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    try:
        model = model_class.from_pretrained(model_name, use_cache=False)
    except TypeError:
        model = model_class.from_pretrained(model_name)
    model.eval()

    # Preprocess inputs
    if seq_len > 0:
        inputs = tokenizer(input_str, return_tensors="pt", padding="max_length", max_length=seq_len)
    else:
        inputs = tokenizer(input_str, return_tensors="pt")
    if model.config.is_encoder_decoder and "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    mask_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    # Run PyTorch inference
    pt_outputs = model(**inputs)
    output_names = list(pt_outputs.keys())
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # print(inputs)
    # print(output_names)
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    # Export and simplify ONNX model
    gen_models_path = f"{model_path}/generatedModels"
    os.makedirs(gen_models_path, exist_ok=True)
    fp32_model_name = export_onnx(model, inputs, output_names, gen_models_path, model_base_name)
    # fp32_model_name = simplify_onnx(gen_models_path, fp32_model_name)

    # overwriting 
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DANGER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # cmd = 'cp -v /local/mnt/workspace/ameynaik/cloud-ai/samples/python/llm/bert-base-cased.onnx /local/mnt/workspace/ameynaik/model_zoo-master-internal-huggingface/bert-base-cased/generatedModels/bert-base-cased_simplified.onnx'
    # os.system(cmd)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DANGER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Run ONNXRT inference
    input_names, ort_outputs = run_model_on_ort(
        f"{gen_models_path}/{fp32_model_name}.onnx", inputs, output_names
    )
    print("PyTorch vs. ONNXRT (MAD):")
    for oname, pto, orto in zip(output_names, pt_outputs.values(), ort_outputs):
        print(oname, np.abs(pto.detach().numpy() - orto).max())
        print(oname, tokenizer.decode(torch.argmax(pto[0, mask_index, :])), tokenizer.decode(torch.argmax(torch.from_numpy(orto)[0, mask_index, :])))

    # Fix onnx for FP16
    fp16_model_name = fix_onnx_fp16(
        inputs, output_names, ort_outputs, gen_models_path, fp32_model_name
    )
    
    # fp16_model_name = fp32_model_name

    print("========================================================================")
    print(f" fp32_model_name = {fp32_model_name} fp16_model_name {fp16_model_name}")
    print("========================================================================")

    # Generate NPI yaml if needed
    # npi_path = f"{model_path}/node_precision_info.yaml"
    # mixed_precision = generate_npi(
    #     original_precision_nodes_pattern, npi_path, gen_models_path, fp16_model_name
    # )

    # print("------------------------------------------------------------------------")
    # print(f"Mixed Precision {mixed_precision} and model_path = {model_path}")
    # print("------------------------------------------------------------------------")
    mixed_precision = False
    # Generate inputFiles
    input_list_file = f"{model_path}/input_list.txt"
    generate_input_files(f"{model_path}/inputFiles", input_names, inputs, input_list_file)

    if not run_on_aic:
        return

    # onnx dynamic shape
    onnx_symbol_defs = {"batch_size": 1, "sequence": inputs["input_ids"].shape[-1]}

    # AICOutputs
    write_output_dir = f"{model_path}/AICOutputs"
    os.makedirs(f"{write_output_dir}/FP32", exist_ok=True)
    os.makedirs(f"{write_output_dir}/FP16", exist_ok=True)

    # Run on AIC in FP32
    assert run_model_on_aic(
        f"{gen_models_path}/{fp32_model_name}.onnx",
        onnx_symbol_defs=onnx_symbol_defs,
        input_list_file=input_list_file,
        convert_to_fp16=False,
        write_output_dir=f"{write_output_dir}/FP32",
    ), "Compilation failed"

    # Run on AIC in FP16
    assert run_model_on_aic(
        f"{gen_models_path}/{fp16_model_name}.onnx",
        onnx_symbol_defs=onnx_symbol_defs,
        input_list_file=input_list_file,
        convert_to_fp16=not mixed_precision,
        node_precision_info=npi_path if mixed_precision else False,
        write_output_dir=f"{write_output_dir}/FP16",
    ), "Compilation failed"

    # Verify outputs
    print("ONNXRT vs. AIC (MAD)")
    for oname, orto in zip(output_names, ort_outputs):
        aico32 = np.fromfile(
            f"{write_output_dir}/FP32/{oname}-activation-0-inf-0.bin", orto.dtype
        ).reshape(orto.shape)
        diff32 = np.abs(orto - aico32).max()
        print(oname, tokenizer.decode(torch.argmax(torch.from_numpy(orto)[0, mask_index, :])), tokenizer.decode(torch.argmax(torch.from_numpy(aico32)[0, mask_index, :])))
        print(oname, "FP32:", diff32)

        aico16 = np.fromfile(
            f"{write_output_dir}/FP16/{oname}-activation-0-inf-0.bin", orto.dtype
        ).reshape(orto.shape)
        print(aico16, orto.dtype, aico16.shape, mask_index)
        print(oname, tokenizer.decode(torch.argmax(torch.from_numpy(orto)[0, mask_index, :])), tokenizer.decode(torch.argmax(torch.from_numpy(aico16)[0, mask_index, :])))
        diff16 = np.abs(orto - aico16).max()
        print(oname, "FP16:", diff16)


def parse_args():
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--model-name",
        required=True,
        help="Model name to generate",
    )
    argp.add_argument(
        "--model-class",
        "--model-type",
        #AMEY default=transformers.AutoModel,
        default=transformers.AutoModelForMaskedLM,
        type=lambda x: getattr(transformers, x),
        help="Model class to use for export, eg: BertForSequenceClassification",
    )
    argp.add_argument(
        "--input-str",
        # default="My name is Sarah and I live in [MASK].",
        default="The man worked as a [MASK].",
        # default="Paris is [MASK] of France.",
        help="Input sentence to encode into tokens",
    )
    argp.add_argument(
        "--seq-len",
        default=128,
        type=int,
        help="Sequence length to encode the input string (0 to disable padding)",
    )
    argp.add_argument(
        "--model-path",
        help="Path to export models, by default base name of the model",
    )
    argp.add_argument(
        "--original-precision-nodes-pattern",
        help="Node names RE pattern to run in FP32",
    )
    # argp.add_argument(
    #     "--unroll-len",
    #     default=5,
    #     type=int,
    #     help="Unroll len of decoder loop"
    # )
    argp.add_argument(
        "--run-on-aic",
        action="store_true",
        help="Run the model on AIC and compare outputs with ONNXRT",
    )
    return argp.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))

