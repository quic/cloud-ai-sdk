from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys
sys.path.append("/opt/qti-aic/examples/apps/qaic-python-sdk")
import qaic
import os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from common_utils import generate_bin
import torch
import onnx
from onnxsim import simplify
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple

torch_inference = False

parser = argparse.ArgumentParser(description='python example.py --model albert-base-v2')

parser.add_argument("--model", required=False, help="select one of the LLMs 'distilbert-base-uncased' 'bert-base-uncased' 'bert-large-uncased' 'bert-base-cased' 'albert-base-v2' 'xlm-roberta-large'", default='albert-base-v2')
args = parser.parse_args()
model = args.model

# model_card = 'xlm-roberta-large'  #RuntimeError: Exporting model exceed maximum protobuf size of 2GB. Please call torch.onnx.export with use_external_data_format=True. FIXME

print(f"Model selected is {model}")

# Sentence examples
sentence = "The man worked as a [MASK]."

def get_example_input(tokenizer):
    max_length = 128
    encodings = tokenizer(sentence, max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
    inputIds = encodings["input_ids"]
    attentionMask = encodings["attention_mask"]
    mask_token_index = torch.where(encodings['input_ids'] == tokenizer.mask_token_id)[1]


    return inputIds, attentionMask, mask_token_index, encodings

torch.manual_seed(10)

model_card = model
output_path = f'{model_card}.onnx'

# Import the pre-trained model
model = AutoModelForMaskedLM.from_pretrained(model_card)

# setup the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_card)

# get input example 
inputIds, attentionMask, mask_token_index, inputs = get_example_input(tokenizer)

pt_outputs = model(**inputs)
output_names = list(pt_outputs.keys())
model_path = model_card
gen_models_path = f"{model_path}/generatedModels"
os.makedirs(gen_models_path, exist_ok=True)
model_base_name = model_card

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


fp32_model_name = export_onnx(model, inputs, output_names, gen_models_path, model_base_name)

if False:
    # get 
    dynamic_dims = {0: 'batch', 1 : 'sequence'}
    dynamic_axes = {
        "input_ids" : dynamic_dims,
        "attention_mask" : dynamic_dims,
        "logits" : dynamic_dims
    }
    input_names = ["input_ids", "attention_mask"]
    inputList = [inputIds, attentionMask]

    torch.onnx.export(
        model,
        args=tuple(inputList),
        f=output_path,
        verbose=False,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        # use_external_data_format=True, # required for model_card = 'xlm-roberta-large'
        opset_version=11,
    )
    print("ONNX Model is being generated successfully for opset Version 11")

    # output_path_simp = output_path.replace(".onnx", "-simp.onnx")
    # if not os.path.exists(output_path_simp):
    model_simp, check = simplify(output_path)
    onnx.save(model_simp, output_path)
    assert check, "Simplified ONNX model could not be validated."

    print("ONNX Model is being simplified and saved successfully.")

if torch_inference:
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(input_ids=inputIds, attention_mask=attentionMask)

    token_logits = model_output.logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_results = torch.topk(mask_token_logits, 5, dim=1)
    print("Model output (top5) from torch-cpu:")
    for i in range(5):
        idx = top_5_results.indices[0].tolist()[i]
        val = top_5_results.values[0].tolist()[i]
        word = tokenizer.decode([idx])
        print(f"{i+1} :(word={word}, index={idx}, logit={round(val,2)})")

from onnx import numpy_helper
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

fp16_model_name = fix_onnx_fp16(inputs, output_names, ort_outputs=None, gen_models_path=gen_models_path, model_base_name=fp32_model_name)

# define the batch_size
batch_size = 1
# Generate binary for QAIC by default the binary is compiled for 1 nsp core, set-size = 10 and fp16 precision.
# qpcPath = generate_bin(onnx_path = output_path) # return path to the folder containing compiled binary. #FIXME: compile cmd is hardcoded.
options_path = f'{model_card}-config.yaml'
# output_path = f'{model_card}.onnx'
print("-----------------------------------------------------------------")
cmd = f'cp -v /local/mnt/workspace/ameynaik/cloud-ai/samples/python/llm/{model_card}/generatedModels/{model_card}_fp16.onnx /local/mnt/workspace/ameynaik/cloud-ai/samples/python/llm/{model_card}.onnx'
# print("overwriting with fix fp16 output.")
# cmd = 'cp -v /local/mnt/workspace/ameynaik/model_zoo-master-internal-huggingface/bert-base-cased/generatedModels/bert-base-cased_fp16.onnx /local/mnt/workspace/ameynaik/cloud-ai/samples/python/llm/bert-base-cased.onnx'
os.system(cmd)
qpcPath = generate_bin(onnx_filename = output_path, yaml_filename=options_path)
print(f'qpcPath : {qpcPath}')
# Compile and load the model
bert_sess = qaic.Session(model_path= qpcPath+'/programqpc.bin', options_path=options_path)

print(bert_sess.model_input_shape_dict.keys(), bert_sess.model_output_shape_dict.keys())
input_shape, input_type = bert_sess.model_input_shape_dict['input_ids']
attn_shape, attn_type = bert_sess.model_input_shape_dict['attention_mask']
token_type_ids_shape, token_type_ids_type = bert_sess.model_input_shape_dict['token_type_ids']
output_shape, output_type = bert_sess.model_output_shape_dict['logits']
print(input_shape, input_type)


input_dict = {"input_ids": inputIds.numpy().astype(input_type), "attention_mask" : attentionMask.numpy().astype(attn_type), "token_type_ids" : inputs['token_type_ids'].numpy().astype(token_type_ids_type)}
output = bert_sess.run(input_dict)

aico16 = np.frombuffer(output['logits'], dtype=output_type).reshape(output_shape)
print(aico16, output_type, output_shape, mask_token_index)
print(tokenizer.decode(torch.argmax(torch.from_numpy(aico16)[0, mask_token_index, :])))

#for oname, orto in zip(output_names, output):
#    print(oname, tokenizer.decode(torch.argmax(torch.from_numpy(aico32)[0, mask_index, :])))
token_logits = np.frombuffer(output['logits'], dtype=output_type).reshape(output_shape) # dtype to be modified based on given model
# token_logits = np.frombuffer(output['logits'], dtype=output_type).reshape(output_shape) # dtype to be modified based on given model
print(token_logits)

mask_token_logits = torch.from_numpy(token_logits[0, mask_token_index, :]).unsqueeze(0)
top_5_results = torch.topk(mask_token_logits, 5, dim=1)
print("Model output (top5) from AIC:")
for i in range(5):
    idx = top_5_results.indices[0].tolist()[i]
    val = top_5_results.values[0].tolist()[i]
    word = tokenizer.decode([idx])
    print(f"{i+1} :(word={word}, index={idx}, logit={round(val,2)})")
