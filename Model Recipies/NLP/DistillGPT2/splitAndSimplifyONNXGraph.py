#!/usr/bin/env python3
#
# Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
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

import onnx
from onnx.tools import update_model_dims
import argparse
from onnxsim import simplify


def SplitGraph(inputFile, outputFile):
    input_names = ['input_ids','position_ids','attention_mask']
    output_names = ['logits']
    onnx.utils.extract_model(inputFile, outputFile, input_names, output_names)
    print("Done with ONNX Model Spliting")

def SimplifyOnnxModel(outputFile):
    input_shapes_list = 'input_ids:64,8 position_ids:64,8 attention_mask:64,8'.split()
    input_shapes = {}
    for x in input_shapes_list:
        if ':' not in x:
            input_shapes[None] = list(map(int, x.split(',')))
        else:
            pieces = x.split(':')
            # for the input name like input:0
            name, shape = ':'.join(
                pieces[:-1]), list(map(int, pieces[-1].split(',')))
            input_shapes.update({name: shape})
    model_onnx, check = simplify(outputFile, input_shapes=input_shapes, dynamic_input_shape=True, skip_shape_inference=False)
    assert check, 'assert check failed'
    onnx.save(model_onnx, outputFile)

def UpdateModelDims(inputFile):
    model = onnx.load(inputFile)
    variable_length_model = update_model_dims.update_inputs_outputs_dims(model, {'image': [1,3,400,400]}, {})

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Split the ONNX Graph")
    parser.add_argument("--onnx_input",
                        type=str,
                        required=True,
                        default="input.onnx",
                        help="ONNX input file")
    parser.add_argument("--onnx_output",
                        type=str,
                        required=True,
                        default="output.onnx",
                        help="ONNX output file to save")
    args = parser.parse_args()
    SplitGraph(args.onnx_input, args.onnx_output)
    SimplifyOnnxModel(args.onnx_output)
    #UpdateModelDims(args.onnx_input)
