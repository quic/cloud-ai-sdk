#Qualcomm Technologies, Inc.Proprietary
#(c) 2020 Qualcomm Technologies, Inc.All rights reserved.
#
#All data and information contained in or disclosed by this document are
#confidential and proprietary information of Qualcomm Technologies, Inc., and
#all rights therein are expressly reserved.By accepting this material, the
#recipient agrees that this materialand the information contained therein
#are held in confidenceand in trustand will not be used, copied, reproduced
#in whole or in part, nor its contents revealed in any manner to others
#without the express written permission of Qualcomm Technologies, Inc.
#!/bin/bash

import onnx_graphsurgeon as gs
import onnx
from collections import OrderedDict
import numpy as np

def remove_node(node):
    for inp in node.inputs:
        inp.outputs.clear()
    # Disconnet input nodes of all output tensors
    for out in node.outputs:
        out.inputs.clear()

@gs.Graph.register()
def replace_with_2d_mask(self, inputs, outputs):
    inp1 = inputs[0]
    inp1.shape = ['batch_size', 'seg_length', 'seg_length']
    inp1.dtype = np.bool
    
    # Disconnect output nodes of all input tensors
    for inp in inputs:
        inp.outputs.clear()
    # Disconnet input nodes of all output tensors
    for out in outputs:
        for out2 in out.outputs:
            output_copy = list(out2.outputs)
            remove_node(out2)
        remove_node(out)
    # Insert the new node.
    return self.layer(op="Unsqueeze", name='', attrs=OrderedDict([('axes', [1])]), inputs=[inp1], outputs=output_copy)

def add_position_input(graphPacked):
    print ("in add pos")
    collectGatherNodes = [node for node in graph.nodes if node.op == "Gather"]
    for gather in collectGatherNodes:
        if gather.inputs[0].name == "bert.embeddings.position_embeddings.weight":
            positionInput = gs.Variable(name="input_position_ids", dtype=np.int64, shape=("batch_size", 'seg_length'))
            print ("shape: ", positionInput)
            gather.inputs[1] = positionInput
            graphPacked.inputs.append(positionInput)
    
    print ("returning")
    return graphPacked

print ("Before model load")
modelFloat = onnx.load("./generatedModels/ONNX/BERT_MLCommons_Flexible_BS_SL.onnx")
print ("after model load")
graph = gs.import_onnx(modelFloat)
print ("after modelfloat")
for node in graph.nodes:
    print ("in For loop")
    if len(node.inputs) > 0 and node.inputs[0].name == "input_mask":
        print ("if cond")
        graph.replace_with_2d_mask(node.inputs, node.outputs)
        print ("replace done")
        break

graph = add_position_input(graph)
print ("Add pos done")
graph.cleanup().toposort()
        
onnx.save(gs.export_onnx(graph), "./generatedModels/ONNX/BERT_MLCommons_Flexible_BS_SL_Packed.onnx")

print ("save done")
