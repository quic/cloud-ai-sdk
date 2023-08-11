import onnx
import numpy as np
from onnx import helper
from onnx import TensorProto
import argparse
import os

## Onnx file is edited to modify input data type from uint8 to float and limit the output tensor size based on postNMSTopN
parser= argparse.ArgumentParser(description='Onnx file edited to modify input data type from float to uint8 and limit the output tensor size based on postNMSTopN')
parser.add_argument('-model', type=str, dest='model', help="Path to onnx model.", required=True)
args, unknown = parser.parse_known_args()

model_def = onnx.load(args.model)
modelname = os.path.splitext(os.path.basename(args.model))[0]
suffix = '_modified.onnx'

## Input data type of uint8 is not supported for now. Changing it to float
model_def.graph.input[0].type.tensor_type.elem_type = 1

out0 = helper.make_tensor_value_info('roi_bbox_nms_out', TensorProto.FLOAT, [100, 4])
out1 = helper.make_tensor_value_info('roi_score_nms_out', TensorProto.FLOAT, [100])
out2 = helper.make_tensor_value_info('roi_class_nms_out', TensorProto.INT32, [100])
out3 = helper.make_tensor_value_info('roi_batch_splits_nms_out', TensorProto.INT32, [1])
out4 = helper.make_tensor_value_info('value_out', TensorProto.FLOAT, [100, 80, 28, 28])

replace_names = {'roi_bbox_nms': 'roi_bbox_nms_out',
                 '569': 'roi_score_nms_out',
                 '568': 'roi_class_nms_out',
                 'roi_batch_splits_nms': 'roi_batch_splits_nms_out',
                 'value': 'value_out'}

for i,n in enumerate(model_def.graph.node):
    for j in range(len(n.output)):
        if n.output[j] in replace_names.keys():
            n.output[j] = replace_names[n.output[j]]
    # Below code is required as 'roi_bbox_nms' goes as input to
    # 'DistributeFPNProposals' node and also to output
    for j in range(len(n.input)):
        if n.input[j] in replace_names.keys():
            n.input[j] = replace_names[n.input[j]]


# Delete existing outputs and add new output tensors
for _ in range(0,4):
    del model_def.graph.output[0]
model_def.graph.output.extend([out0, out1, out2, out3, out4])
print("Writing model to ", modelname+suffix)
onnx.save(model_def,modelname+suffix)
