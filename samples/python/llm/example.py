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


    return inputIds, attentionMask, mask_token_index

torch.manual_seed(10)

model_card = model
output_path = f'{model_card}.onnx'

# Import the pre-trained model
model = AutoModelForMaskedLM.from_pretrained(model_card)

# setup the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_card)

# get input example 
inputIds, attentionMask, mask_token_index = get_example_input(tokenizer)

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

# define the batch_size
batch_size = 1
# Generate binary for QAIC by default the binary is compiled for 1 nsp core, set-size = 10 and fp16 precision.
# qpcPath = generate_bin(onnx_path = output_path) # return path to the folder containing compiled binary. #FIXME: compile cmd is hardcoded.
options_path = f'{model_card}-config.yaml'
qpcPath = generate_bin(onnx_filename = output_path, yaml_filename=options_path)

# Compile and load the model
bert_sess = qaic.Session(model_path= qpcPath+'/programqpc.bin', options_path=options_path)
input_shape, input_type = bert_sess.model_input_shape_dict['input_ids']
attn_shape, attn_type = bert_sess.model_input_shape_dict['attention_mask']
output_shape, output_type = bert_sess.model_output_shape_dict['logits']


input_dict = {"input_ids": inputIds.numpy().astype(input_type), "attention_mask" : attentionMask.numpy().astype(attn_type)}
output = bert_sess.run(input_dict)
token_logits = np.frombuffer(output['logits'], dtype=output_type).reshape(output_shape) # dtype to be modified based on given model
print(token_logits)

mask_token_logits = torch.from_numpy(token_logits[0, mask_token_index, :]).unsqueeze(0)
top_5_results = torch.topk(mask_token_logits, 5, dim=1)
print("Model output (top5) from torch-cpu:")
for i in range(5):
    idx = top_5_results.indices[0].tolist()[i]
    val = top_5_results.values[0].tolist()[i]
    word = tokenizer.decode([idx])
    print(f"{i+1} :(word={word}, index={idx}, logit={round(val,2)})")
