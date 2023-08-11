"""
##############################################################################
#
#Copyright (c) 2021 Qualcomm Technologies, Inc.
#All Rights Reserved.
#Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#All data and information contained in or disclosed by this document are
#confidential and proprietary information of Qualcomm Technologies, Inc., and
#all rights therein are expressly reserved. By accepting this material, the
#recipient agrees that this material and the information contained therein
#are held in confidence and in trust and will not be used, copied, reproduced
#in whole or in part, nor its contents revealed in any manner to others
#without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################
"""
from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
from transformers import AutoConfig
import torch
import os
from transformers import AutoTokenizer
from pdb import set_trace as bp
import onnxruntime as onnxrt
import argparse
import onnx
import shutil 

torch.random.manual_seed = 123

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    #okenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def get_example_inputs(prompt_text, hidden_size, num_layer, num_attention_heads, seq_len=8, past_seq_len = 1):    
    tokenizer = get_tokenizer()
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)
    encodings_dict['input_ids'] = [ids[:seq_len] for ids in encodings_dict['input_ids']]
    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)
    position_ids = (attention_mask.long().cumsum(-1) - 1)[:,:seq_len]
    position_ids.masked_fill_(position_ids < 0, 0)
    #Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, past_seq_len, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.rand(past_shape).type(torch.float32).to(device)) 
    return input_ids, attention_mask, position_ids, empty_past

def get_inputs(prompt_text):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    num_attention_heads = config.n_head
    hidden_size = config.n_embd
    num_layer = config.n_layer
    input_ids, attention_mask, position_ids, empty_past = get_example_inputs(prompt_text, hidden_size, num_layer, num_attention_heads)
    print("input_ids", input_ids.shape)
    print("attention_mask", attention_mask.shape)
    print("position_ids", position_ids.shape)
    print("past0", empty_past[0].shape)
    return input_ids, attention_mask, position_ids, empty_past, num_layer

def get_model():
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = MyGPT2LMHeadModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    #model = GPT2LMHeadModel_BeamSearchStep.from_pretrained(model_name, config=config, batch_size=1, beam_size=4, cache_dir=cache_dir)
    model.eval().to(device)
    return model

def save_inputs_outputs(prompt_text, onnx_model_path):
    input_ids, attention_mask, position_ids, empty_past, num_layer = get_inputs(prompt_text)
    past_names = [f'past_{i}' for i in range(num_layer)]
    inputs_dir ='./inputFiles/'
    if not os.path.isdir(inputs_dir):
        os.makedirs(inputs_dir)
    ort_inputs = {'input_ids': input_ids.numpy(), 'attention_mask': attention_mask.numpy(), 'position_ids': position_ids.numpy()}
    for idx in range(len(past_names)):
        ort_inputs[past_names[idx]] = empty_past[idx].numpy()
    for inp in ort_inputs:
        ort_inputs[inp].tofile(inputs_dir + inp+'.raw')
    sess = onnxrt.InferenceSession(onnx_model_path)
    ort_outputs = sess.run(None, ort_inputs)
    outputs_dir = 'OnnxRTOutputs/'
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir)    
    present_names = [f'present_{i}' for i in range(num_layer)]
    output_names = ['logits']+present_names
    for idx in range(len(ort_outputs)):
        ort_outputs[idx].tofile(outputs_dir + output_names[idx]+'.raw')

def save_onnx(prompt_text, onnx_model_path):
    output_dir = os.path.dirname(onnx_model_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    model = get_model()
    if model_name in ['gpt2-large','gpt2-xl']:
        out_dir = os.path.dirname(onnx_model_path)
        out_path = os.path.basename(onnx_model_path)
        out_dir = out_dir + '/' + model_name.replace('-','_')
        os.makedirs(out_dir, exist_ok=True)
        onnx_model_path =  os.path.join(out_dir, out_path)
        Gpt2Helper.export_onnx(model, device, onnx_model_path, use_external_data_format=True)
        model = onnx.load(onnx_model_path)
        shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        onnx.save_model(model, onnx_model_path, save_as_external_data=True, all_tensors_to_one_file=True,location=out_path.split('.')[0]+'.bin')
    else:
        Gpt2Helper.export_onnx(model, device, onnx_model_path)  
        

def save_torchscript(prompt_text, torch_jit_path):
    output_dir = os.path.dirname(torch_jit_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    model = get_model()
    input_ids, attention_mask, position_ids, empty_past, _ = get_inputs(prompt_text)
    traced_script_module = torch.jit.trace(model, (input_ids,position_ids,attention_mask,*empty_past))
    # # Save the TorchScript model
    traced_script_module.save(torch_jit_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
		description='Generate GPT2 ONNX,torchscript model and save sample inputs and reference onnxrt outputs as raw files')
    parser.add_argument("--model-name",
                        required=True,
                        dest='model_name',
                        help="name of the model [gpt2,gpt2-medium,gpt2-large,gpt2-xl]")
    parser.add_argument("--output-path",
                        required=False,
                        dest='output_path',
                        help="Path for output Model")
    parser.add_argument("--save-torchscript",
                        action='store_true',
                        required=False,
                        help="Save Torchscipt Model")
    parser.add_argument("--save_raw_files",
                        action='store_true',
                        required=False,
                        help="Save Sample inputs and onnxrt outputs as raw files")
    args = parser.parse_args()
    cache_dir = os.path.join(".", "cache_models")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    EXAMPLE_Text = ['here is an example of gpt2 model']
    device = torch.device("cpu")
    if args.output_path is None:
        output_path = './'+args.model_name + '.onnx'
    else:
        output_path = args.output_path
    onnx_model_path = output_path
    torch_jit_path = output_path
    model_name = args.model_name
    if args.save_torchscript: 
        save_torchscript(EXAMPLE_Text, torch_jit_path)
    elif args.save_raw_files:
        save_inputs_outputs(EXAMPLE_Text, onnx_model_path)
    else:
        save_onnx(EXAMPLE_Text, onnx_model_path)
