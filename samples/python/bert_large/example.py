from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys
sys.path.append("/opt/qti-aic/examples/apps/qaic-python-sdk")
import qaic
import os
import torch
import onnx
from onnxsim import simplify
from argparse import ArgumentParser
import numpy as np

# Sentence examples
sentence = "The man worked as a [MASK]."
def get_example_input(tokenizer):
    encodings = tokenizer(sentence, return_tensors='pt')
    inputIds = encodings["input_ids"]
    attentionMask = encodings["attention_mask"]
    mask_token_index = torch.where(encodings['input_ids'] == tokenizer.mask_token_id)[1]


    return inputIds, attentionMask, mask_token_index

def generate_bin(onnx_path, batch_size=1, aic_num_cores=1, precision='fp16'):
    filename, extension = os.path.splitext(onnx_path)
    onnx_folder = os.path.dirname(onnx_path)
    qpc_bin = onnx_folder+filename+'_qpc'
    if os.path.isdir(qpc_bin):
        cmd = f'sudo rm -fr {qpc_bin}'
        os.system(cmd)
        print(f'Removing existing QPC')

    # cmd = f'/opt/qti-aic/exec/qaic-exec -m={onnx_path} -aic-hw -aic-hw-version=2.0 -convert-to-{precision} -onnx-define-symbol=batch_size,{batch_size} -aic-num-cores={aic_num_cores}  -aic-binary-dir={qpc_bin}'
    
    if False:
        cmd = f'/opt/qti-aic/exec/qaic-exec \
              -m=./bert-large-uncased.onnx \
              -onnx-define-symbol=batch,1 \
              -onnx-define-symbol=sequence,9 \
              -aic-hw -aic-hw-version=2.0 \
              -aic-num-cores=4 \
              -ols=1 \
              -mos=1 \
              -convert-to-fp16 \
              -aic-num-of-instances=7 \
              -aic-binary-dir={qpc_bin}'
    else:
        cmd = f'/opt/qti-aic/exec/qaic-exec \
            -m=./bert-large-uncased.onnx \
            -onnx-define-symbol=batch,1 \
            -onnx-define-symbol=sequence,9 \
            -aic-hw -aic-hw-version=2.0 \
            -aic-num-cores=4 \
            -ols=1 \
            -mos=1 \
            -aic-num-of-instances=1 \
            -aic-binary-dir={qpc_bin}'
        
    os.system(cmd)
    print(f'Running : {cmd}')

    return qpc_bin

torch.manual_seed(10)

model_card = 'bert-large-uncased'
output_path = f'{model_card}.onnx'

# Import the pre-trained model
model = AutoModelForMaskedLM.from_pretrained(model_card)

tokenizer = AutoTokenizer.from_pretrained(model_card)

inputIds, attentionMask, mask_token_index = get_example_input(tokenizer)

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
    opset_version=11,
)
print("ONNX Model is being generated successfully for opset Version 11")

# output_path_simp = output_path.replace(".onnx", "-simp.onnx")
# if not os.path.exists(output_path_simp):
model_simp, check = simplify(
    output_path#,
    # input_shapes={"input_ids" : [1, 9], "attention_mask" : [1, 9]}, #FIXME: why?
    # dynamic_input_shape=True,
)
onnx.save(model_simp, output_path)
assert check, "Simplified ONNX model could not be validated."

print("ONNX Model is being simplified and saved successfully.")

if False:
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
qpcPath = generate_bin(onnx_path = output_path) # return path to the folder containing compiled binary. #FIXME: compile cmd is hardcoded.
#FIXME: read yaml to generate binary?

# Compile and load the model
bert_sess = qaic.Session(model_path= qpcPath+'/programqpc.bin', options_path='./bert_large_config.yaml')
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

'''
TODO:
1. resolve FIXMEs
2. currently the model is compiled for sequence length of 9. better to compile it for sequence len of 128 and mask the input accordingly.
3. currently the performance is bad with fp16, hence model is compiled with fp32.
4. generate_bin needs to be changed to use .yaml file.
'''
