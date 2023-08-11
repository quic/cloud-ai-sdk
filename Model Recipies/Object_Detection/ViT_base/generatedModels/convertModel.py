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
import sys
import onnx
import torch
from onnxsim import simplify
from transformers import ViTForImageClassification

assert int(sys.argv[1]) in [224, 384], "Input size should be 224 or 384."
model = ViTForImageClassification.from_pretrained(f'google/vit-base-patch16-{int(sys.argv[1])}')
# model.eval()     # No need to add this as the model is in eval mode by default


os.makedirs("./ONNX/", exist_ok=True)
os.makedirs("./Pytorch/", exist_ok=True)


# onnx_op_path = f"./ONNX/vit_base_16_{int(sys.argv[1])}.onnx"
onnx_op_path = f"./vit_base_16_{int(sys.argv[1])}.onnx"
dummy_input = torch.randn(1, 3, int(sys.argv[1]), int(sys.argv[1])).type(torch.FloatTensor).to('cpu')
torch.onnx.export(model, dummy_input, onnx_op_path, input_names=['image'], output_names=['output'], dynamic_axes={'image' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})

onnx_model = onnx.load(onnx_op_path)
onnx_model_simp, check = simplify(onnx_model, input_shapes={'image' : [1, 3, int(sys.argv[1]), int(sys.argv[1])]}, dynamic_input_shape=True)
onnx.save(onnx_model_simp, onnx_op_path)
print("ONNX model saved at: ", onnx_op_path)


# torch_op_path = f"./Pytorch/vit_base_16_{int(sys.argv[1])}.pt"
torch_op_path = f"./vit_base_16_{int(sys.argv[1])}.pt"
traced_model = torch.jit.trace(model, dummy_input, strict=False)
# Using strict as false, as model is returning a dict instead of tensor.
traced_model.save(torch_op_path)
print("Torchscript model saved at: ", torch_op_path)