#!/usr/bin/env python3
#
# Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
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
import time
import onnx
import torch
import argparse
import numpy as np
import torch.nn as nn
from onnxsim import simplify
import onnx_graphsurgeon as gs
sys.path.append('./yolov5')


import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU


# python createModels.py --model_type 5s --op_folder ./ONNX/ --model_with_nms \
# --use_qnms False --max_total_size_quantity 100 --iou_thresh 0.65 \
# --score_thresh 0.001 --img_h 320 --img_w 416

# 1a. With ABP QNMS - For PyTorch model - This would use PyTorch Custom op
#     Case-2 of QNMS
# python createModels.py --model_type 5s --op_folder ./ --model_with_qnms \
# --use_pytorch_custom_op --iou_thresh 0.65 --score_thresh 0.001 \
# --img_h 640 --img_w 640

# 1b. With ABP QNMS - For ONNX model - This would stitch QNMS node in exported
#     ONNX model - Case-1 of QNMS
# python createModels.py --model_type 5s --op_folder ./ --model_with_qnms \
# --iou_thresh 0.65 --score_thresh 0.001 --img_h 640 --img_w 640

# 2. Without ABP NMS - For both PyTorch and ONNX model
# python createModels.py --model_type 5s --op_folder ./ --img_h 640 --img_w 640


def addCustomPlugin(ori_model_path, new_model_path, input_shapes, \
                    max_output_bboxes_per_class, max_total_size_quantity, \
                    iou_thresh, score_thresh):

    simplifiedModel, check = simplify(ori_model_path, \
                        input_shapes=input_shapes, dynamic_input_shape=True)
    assert check, 'assert check failed'
    graph = gs.import_onnx(simplifiedModel)
    graph.fold_constants()
    graph.cleanup().toposort()


    maxOutputSizePerClass =  gs.Constant(name="max_output_size_per_class", \
                                         values=np.array(\
                                                max_output_bboxes_per_class, \
                                                dtype="int32"))
    maxTotalSize =  gs.Constant(name="max_total_size", \
                                values=np.array(\
                                            max_total_size_quantity, \
                                            dtype="int32"))
    iouThreshold = gs.Constant(name="iou_threshold", \
                               values=np.array(iou_thresh, dtype="float32"))
    # float(0.65)
    scoreThreshold = gs.Constant(name="score_threshold",
                                 values=np.array(score_thresh, dtype="float32"))
    # float(0.001)

    nmsOp = "CustomQnmsYolo"
    nmsAttrs = {
        'pad_per_class' : 0,
        'clip_boxes' : 0
    }

    # NMS Outputs
    batchSize = graph.inputs[0].shape[0]
    numOutputNumDetections = gs.Variable(name="num_detections", \
                                         dtype=np.int32, shape=[batchSize])
    numOutputBoxes = gs.Variable(name="detection_boxes", dtype=np.float32, \
                                 shape=[batchSize, max_total_size_quantity, 4])
    numOutputScores = gs.Variable(name="detection_scores", dtype=np.float32, \
                                  shape=[batchSize, max_total_size_quantity])
    numOutputsClasses = gs.Variable(name="detection_classes", dtype=np.int32, \
                                    shape=[batchSize, max_total_size_quantity])

    nmsOutputs = [numOutputBoxes, numOutputScores, numOutputsClasses, \
                  numOutputNumDetections]
    qNMS = gs.Node(op=nmsOp, name="NonMaxSuppression", attrs=nmsAttrs)

    graph.nodes.append(qNMS)

    print("Original Graph output shapes: ")
    print(graph.outputs[0].shape)
    print(graph.outputs[1].shape)

    qNMS.inputs.append(graph.outputs[0])
    qNMS.inputs.append(graph.outputs[1])
    qNMS.inputs.append(maxOutputSizePerClass)
    qNMS.inputs.append(maxTotalSize)
    qNMS.inputs.append(iouThreshold)
    qNMS.inputs.append(scoreThreshold)

    qNMS.outputs = nmsOutputs
    graph.outputs = nmsOutputs
    graph.cleanup().toposort()
    exportGraph = gs.export_onnx(graph)

    for onnx_node in exportGraph.graph.node:
        if onnx_node.op_type == "CustomQnmsYolo":
            onnx_node.domain = 'com.qti.aisw.onnx'

    defaultOpsetVersion = exportGraph.opset_import[0].version
    qaicOpset = onnx.helper.make_operatorsetid("com.qti.aisw.onnx", defaultOpsetVersion)
    exportGraph.opset_import.append(qaicOpset)
    onnx.checker.check_model(exportGraph)
    onnx.save(exportGraph, new_model_path)
    print(f'ONNX QNMS model export success, saved as {new_model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--op_folder', type=str, default='./', \
                            help='Output folder path to save model')
    parser.add_argument('--model_type', type=str, default='5s', \
                            help='Yolov5 model variant. Should be from [5s, 5m, 5l]')
    parser.add_argument('--model_with_qnms', dest='model_with_qnms', \
                            action='store_true', \
                            help='Use QNMS as a part of model in pytorch.')
    parser.add_argument('--use_pytorch_custom_op', \
                            dest='use_pytorch_custom_op', \
                            action='store_true', \
                            help='Export model with QNMS custom op from \
                                Pytorch. (If enabled then this would simulate \
                                Case-2 of QNMS or else Case-1 of QNMS. \
                                If this is not used then case-1 is triggered \
                                and ONNX model is saved.')
    parser.add_argument('--max_output_boxes_per_class', type=int, default=26, \
                            help='Max number of bboxes each class can have')
    parser.add_argument('--max_total_size_quantity', type=int, default=2080, \
                            help='Maximum bounding box from image across all \
                                 the classes')
    parser.add_argument('--iou_thresh', type=float, default=0.65, \
                            help='IOU Threshold')
    parser.add_argument('--score_thresh', type=float, default=0.001, \
                            help='Score Threshold')
    parser.add_argument('--clip_boxes', dest='clip_boxes', \
                            action='store_true', help='Clip boxes in NMS.')
    parser.add_argument('--pad_per_class', dest='pad_per_class', \
                            action='store_true', \
                            help='Padding applied to make the result tensor \
                                have max_total_size_quantity size.')
    parser.add_argument('--img_h', type=int, default=640, help='Image Height')
    parser.add_argument('--img_w', type=int, default=640, help='Image Width')
    args = parser.parse_args()

    assert args.model_type in ['5s', '5m', '5l']

    weights_path = args.op_folder + f"/yolov{args.model_type}.pt"

    onnx_op_folder = args.op_folder + "/ONNX/"
    os.makedirs(onnx_op_folder, exist_ok=True)
    torch_op_folder = args.op_folder + "/PyTorch/"
    os.makedirs(torch_op_folder, exist_ok=True)


    if args.model_with_qnms:
        onnx_op_path = onnx_op_folder + \
                            f"/yolov{args.model_type}_{args.img_h}_" + \
                                f"{args.img_w}_with_abp_qnms.onnx"
        torch_op_path = torch_op_folder + \
                            f"/yolov{args.model_type}_{args.img_h}_" + \
                                f"{args.img_w}_with_abp_qnms.pt"
    else:
        onnx_op_path = onnx_op_folder + \
                            f"/yolov{args.model_type}_{args.img_h}_" + \
                                f"{args.img_w}_without_abp_nms.onnx"
        torch_op_path = torch_op_folder + \
                            f"/yolov{args.model_type}_{args.img_h}_" + \
                                f"{args.img_w}_without_abp_nms.pt"

    model = attempt_load(weights_path, map_location=torch.device('cpu'), \
                                inplace=False)  # load FP32 model

    model.set_export_params(apply_nms=args.model_with_qnms, \
                            use_pytorch_custom_op=\
                                args.use_pytorch_custom_op, \
                            max_output_boxes_per_class=\
                                args.max_output_boxes_per_class, \
                            iou_thresh=args.iou_thresh, \
                            score_thresh=args.score_thresh, \
                            clip_boxes=args.clip_boxes, \
                            pad_per_class=args.pad_per_class)

    labels = model.names

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  #  export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)


    dummy_input = torch.randn(1, 3, args.img_h, args.img_w)
    model.eval()


    if args.model_with_qnms:
        if args.use_pytorch_custom_op:
            # Export torchscript model only if using pytorch custom op for QNMS
            traced_model = torch.jit.trace(model, dummy_input, \
                                            check_trace=False, strict=False)
            traced_model.save(torch_op_path)
            print(f"Torchscript model saved: {torch_op_path}")
        else:
            # ONNX export
            torch.onnx.export(model, dummy_input, onnx_op_path, \
                                opset_version=11, input_names=['images'], \
                                dynamic_axes={'images' : {0 : 'batch_size'}})

            # Check model if Case-1 of QNMS is triggered. In case-2, we cant
            # check or simplify the model due to QNMS node present in the graph.
            onnx_model = onnx.load(onnx_op_path)
            model_simp, check = simplify(onnx_model, \
                                         input_shapes={'images' : \
                                             [1, 3, args.img_h, args.img_w]}, \
                                         dynamic_input_shape=True)
            onnx.save(model_simp, onnx_op_path)
            onnx.checker.check_model(onnx_model)

            # Below lines are for case-1 of QNMS
            # Till now model contains Feature extraction and ABP part,
            # QNMS node will be stitched in below function
            addCustomPlugin(onnx_op_path, onnx_op_path, \
                            {'images' : [1, 3, args.img_h, args.img_w]}, \
                            args.max_output_boxes_per_class, \
                            args.max_total_size_quantity, args.iou_thresh, \
                            args.score_thresh)
            print(f"ONNX model saved: {onnx_op_path}")

    else:
        # Export torchscript
        traced_model = torch.jit.trace(model, dummy_input, check_trace=False, \
                                        strict=False)
        traced_model.save(torch_op_path)
        print(f"Torchscript model saved: {torch_op_path}")

        # ONNX export
        torch.onnx.export(model, dummy_input, onnx_op_path, opset_version=11, \
                            input_names=['images'], \
                            output_names=['feature_map_1', 'feature_map_2', \
                                          'feature_map_3'], \
                            dynamic_axes={'images' : {0 : 'batch_size'}, \
                                        'feature_map_1' : {0 : 'batch_size'}, \
                                        'feature_map_2' : {0 : 'batch_size'}, \
                                        'feature_map_3' : {0 : 'batch_size'}})

        # Simplify the model
        onnx_model = onnx.load(onnx_op_path)
        model_simp, check = simplify(onnx_model, \
                                     input_shapes={'images' : \
                                         [1, 3, args.img_h, args.img_w]}, \
                                     dynamic_input_shape=True)
        onnx.save(model_simp, onnx_op_path)
        onnx.checker.check_model(onnx_model)

        print(f"ONNX model saved: {onnx_op_path}")

