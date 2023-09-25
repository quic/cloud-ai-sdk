'''
Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import sys
sys.path.append("/opt/qti-aic/examples/apps/qaic-python-sdk")
import qaic
import numpy as np
import torchvision
import torch
import pandas as pd
import os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from common_utils import generate_bin
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import onnx
from onnxsim import simplify

image_size = 224

model_name = f'vit-base-patch16-{image_size}'

# Import the model
model = ViTForImageClassification.from_pretrained(f'google/{model_name}')
onnx_filename = f'{model_name}.onnx'

# Export the PyTorch model to ONNX
dummy_input = torch.randn(1, 3, image_size, image_size).type(torch.FloatTensor)
torch.onnx.export(model,                # PyTorch model
                  dummy_input,           # Input tensor
                  onnx_filename,          # Output file
                  export_params=True,   # Export the model parameters
                  opset_version=11,     # ONNX opset version
                  do_constant_folding=True,  # Fold constant values for optimization
                  input_names=['image'],    # Input tensor names
                  output_names=['output'],  # Output tensor names
                  dynamic_axes={'image': {0: 'batch_size'},  # Dynamic axes
                                'output': {0: 'batch_size'}})

# apply onnxsim (optional)
onnx_model = onnx.load(onnx_filename)
onnx_model_simp, check = simplify(onnx_model)
onnx.save(onnx_model_simp, onnx_filename)
print("ONNX model saved at: ", onnx_filename)

# Generate binary for QAIC by default the binary using a helper library. 
qpcPath = generate_bin(onnx_filename = onnx_filename , yaml_filename ='./vit_config.yaml') # return path to the folder containing compiled binary. 

# Compile and load the model
vit_sess = qaic.Session(model_path= qpcPath+'/programqpc.bin', options_path='./vit_config.yaml')
vit_sess.setup() 
input_shape, input_type = vit_sess.model_input_shape_dict['image']
output_shape, output_type = vit_sess.model_output_shape_dict['output']

processor = ViTImageProcessor.from_pretrained(f'google/{model_name}')

# input sample
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

device = True
if device:
    print("INFO: running inference on Qualcomm Cloud AI 100")
    input_data = inputs['pixel_values'].numpy().astype(input_type)
    input_dict = {'image': input_data}
    output = vit_sess.run(input_dict)
    logits = np.frombuffer(output['output'], dtype=output_type).reshape(output_shape) # dtype to be modified based on given model
else:
    print("INFO: running inference on CPU")
    outputs = model(**inputs)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])