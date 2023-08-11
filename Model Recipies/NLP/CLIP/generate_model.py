##############################################################################
#
#Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
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
import os
import onnx
import torch
import numpy as np
from onnxsim import simplify
from transformers import CLIPModel
import argparse

def generate_inputs():
    input_ids = torch.from_numpy(np.fromfile("./input_ids.raw", dtype=np.int64).reshape(2,7))
    attention_mask = torch.from_numpy(np.fromfile("./attention_mask.raw", dtype=np.int64).reshape(2,7))
    pixel_values  = torch.from_numpy(np.fromfile("./pixel_values.raw", dtype=np.float32).reshape(4, 3, 224, 224))
    causal_attention_mask  = torch.from_numpy(np.fromfile("./causal_attention_mask.raw", dtype=np.float32).reshape(2,1,7,7))
    class_embeds = torch.from_numpy(np.fromfile("./class_embeds.raw", dtype=np.float32).reshape(4,1,768))
    input = (input_ids,pixel_values,attention_mask,causal_attention_mask,class_embeds)
    return input

def save_onnx(onnx_op_path, model, input):
    torch.onnx.export(model, input, onnx_op_path,
                    opset_version=11,
                    input_names=['input_ids','pixel_values','attention_mask','causal_attention_mask','class_embeds'],
                    output_names=['logits_per_image', 'logits_per_text'],
                    dynamic_axes={'input_ids' : {0 : 'Num_Class', 1: 'Sequence_len'},
                                    'pixel_values' : {0 : 'Batch_Size'},
                                    'attention_mask' : {0 : 'Num_Class', 1: 'Sequence_len'},
                                    'causal_attention_mask' : {0 : 'Num_Class', 2: 'Sequence_len', 3: 'Sequence_len'},
                                    'class_embeds': {0 : 'Batch_Size'}},
                    do_constant_folding=False)
    print("ONNX model saved at: ", onnx_op_path)
    return None

def SplitGraph(inputFile, outputFile, input_names, output_names):
    onnx.utils.extract_model(inputFile, outputFile, input_names, output_names)
    print("Done with ONNX Model Spliting")

def save_split_onnx(onnx_op_path, input_names, output_names, onnx_split_op_path):
    inputFile = onnx_op_path
    outputFile = onnx_split_op_path
    SplitGraph(inputFile, outputFile, input_names, output_names)
    return None

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="CLIP Model Generation and Splitting")
    parser.add_argument("--save_split_onnx",
                        action="store_true",
                        required=False,
                        help="Save ONNX file with two outputs [logits per image, logits per text]")
    args = parser.parse_args()

    #Set Variables
    os.makedirs("./ONNX/", exist_ok=True)
    os.makedirs("./Pytorch/", exist_ok=True)
    onnx_op_path = "./ONNX/clip-vit-base-patch16.onnx"
    #IF Split required
    onnx_split_op_path = "./ONNX/clip-vit-base-patch16_split.onnx"

    #Required for model splitting
    input_names = ['input_ids','pixel_values','attention_mask', 'causal_attention_mask', 'class_embeds']
    output_names = ['logits_per_image', 'logits_per_text']

    #Model Object
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    #Generate Input Tuple
    input = generate_inputs()

    #Saving ONNX File
    save_onnx(onnx_op_path, model, input)
    if args.save_split_onnx:
        save_split_onnx(onnx_op_path, input_names, output_names, onnx_split_op_path)
