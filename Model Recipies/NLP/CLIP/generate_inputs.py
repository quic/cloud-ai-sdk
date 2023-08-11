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
import torch
import argparse
from PIL import Image
import requests
from transformers import CLIPProcessor
import numpy as np
import torch

import os
os.environ['TRANSFORMERS_CACHE'] = './cache/'

def generate_inputs(num_class,batch_size):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    #----------------Text
    text = []
    base_text=["a photo of a cat", "a photo of a dog"]
    base_len = len(base_text)
    if num_class <= 0:
        print("num_class should be greater than 0.")
        return None
    elif num_class == 1:
        text = [base_text[0]]
    else:
        if num_class%base_len == 0:
            for i in range(num_class//base_len):
                text.append(base_text[0])
                text.append(base_text[1])
        else:
            for i in range(num_class//base_len):
                text.append(base_text[0])
                text.append(base_text[1])
            text.append(base_text[0])
    #-----------------Images
    images=[]
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    for i in range(batch_size):
        images.append(image)
    img_tuple = tuple(images)
    #---------------Processor
    inputs = processor(text=text, images=img_tuple, return_tensors="pt", padding=True)
    seq_len = inputs['input_ids'].shape[1]
    causal_attention_mask = generate_causal_attention_mask(num_class,seq_len)
    class_embeds = generate_class_embeds(batch_size)
    inputs['causal_attention_mask'] = causal_attention_mask
    inputs['class_embeds'] = class_embeds
    for key in inputs.keys():
        print(key, " :", inputs[key].shape,inputs[key].dtype)
        inputs[key].detach().numpy().tofile(key+".raw")
    return None

def generate_causal_attention_mask(nc,seq_len):
    mask = torch.empty(nc, seq_len, seq_len)
    mask.fill_(float("-inf"))
    print(mask.dtype)
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask #mask.numpy().tofile("./generated_triu_mask_2_1_7_7.raw")

def generate_class_embeds(bs):
    # Generate a tensor with random values drawn from a normal distribution
    torch.random.manual_seed(123)
    class_embeds = torch.randn(bs,1,768)
    return class_embeds #class_embeds.numpy().tofile("./class_embed.raw")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Generate CLIP Inputs")
    parser.add_argument("--num_class",
                        type=int,
                        required=True,
                        default=2,
                        help="Number of sentences in input")
    parser.add_argument("--batch_size",
                        type=int,
                        required=True,
                        default=1,
                        help="Number of images in input")
    args = parser.parse_args()
    generate_inputs(args.num_class, args.batch_size)
