#!/usr/bin/env python3
#
# Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.

import os
import sys
import onnx
import torch
import argparse
from onnxsim import simplify    
sys.path.append('./yolov3')

from models.experimental import attempt_load


# python createModels.py --op_folder ../modelFiles/ --img_h 320 --img_w 416
# python createModels.py --op_folder ../modelFiles/ --img_h 416 --img_w 416
# python createModels.py --op_folder ../modelFiles/ --img_h 608 --img_w 608
# python createModels.py --op_folder ../modelFiles/ --img_h 608 --img_w 800
# python createModels.py --op_folder ../modelFiles/ --img_h 640 --img_w 640
# python createModels.py --op_folder ../modelFiles/ --img_h 640 --img_w 1152

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--op_folder', type=str, default='../modelFiles/', 
                                        help='Output folder path to save model')
    parser.add_argument('--img_h', type=int, default=640, help='Image Height')
    parser.add_argument('--img_w', type=int, default=640, help='Image Width')
    args = parser.parse_args()
    
    weights_path = "./yolov3.pt"
    onnx_op_folder = args.op_folder + "/ONNX/"
    torch_op_folder = args.op_folder + "/PYTORCH/"
    os.makedirs(onnx_op_folder, exist_ok=True)
    os.makedirs(torch_op_folder, exist_ok=True)
    onnx_op_path = onnx_op_folder + f"/yolov3_{args.img_h}_{args.img_w}" + \
                                        "_without_abp_nms.onnx"
    torch_op_path = torch_op_folder + f"/yolov3_{args.img_h}_{args.img_w}" + \
                                        "_without_abp_nms.pt"

    model = attempt_load(weights_path, map_location=torch.device('cpu'), 
                            inplace=False)  # load FP32 model
    labels = model.names
    
    #######################   Without NMS    ############################
    dummy_input = torch.randn(1, 3, args.img_h, args.img_w)
    model.eval()
    op = model(dummy_input)
    
    # Torchscript model export
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(torch_op_path)
    print(f'Torchscript model export success, saved as {torch_op_path}')

    # ONNX export
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    torch.onnx.export(model, dummy_input, onnx_op_path, opset_version=11, 
                            input_names=['images'], 
                            output_names=['feature_map_1', 'feature_map_2', \
                                          'feature_map_3'], \
                            dynamic_axes={'images' : {0 : 'batch_size'}, \
                                        'feature_map_1' : {0 : 'batch_size'}, \
                                        'feature_map_2' : {0 : 'batch_size'}, \
                                        'feature_map_3' : {0 : 'batch_size'}})
    
    # Checks
    onnx_model = onnx.load(onnx_op_path)  # load onnx model
    model_simp, check = simplify(onnx_model, \
                                 input_shapes={'images' : \
                                     [1, 3, args.img_h, args.img_w]}, \
                                 dynamic_input_shape=True)
    onnx.save(model_simp, onnx_op_path)
    onnx.checker.check_model(onnx_model)  # check onnx model
    print(f'ONNX export success, saved as {onnx_op_path}')
