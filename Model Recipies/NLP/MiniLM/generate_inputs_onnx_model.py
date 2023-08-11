import onnxruntime
import onnx
import numpy as np
from pdb import set_trace as bp
import sys
import argparse
import os
np.random.seed(123)

def generate_inputs(onnx_model_path, batch_size,sequence=128):
    batch_size = int(batch_size)
    sequence = int(sequence)
    Num_Class = 2
    elem_types = {1:np.float32, 7:np.int64, 9:bool, 6:np.int32,2:np.uint8}
    dims_map = {'batch':batch_size, 'src_seq_len':sequence, 'height': 32, 'width':768, 'src_len':sequence,'sequence':sequence}
    model = onnx.load(onnx_model_path)
    ort_inputs = {}
    for inp in model.graph.input:
        shape = [i.dim_value if i.dim_value!=0 else dims_map[i.dim_param] for i in inp.type.tensor_type.shape.dim]
        if inp.name=='input_ids':
            ort_inputs[inp.name] = np.random.randint(0,25000,shape).astype(elem_types[inp.type.tensor_type.elem_type]).reshape(shape)
        elif inp.name=='position_ids':
            ort_inputs[inp.name] = np.arange(shape[-1]).astype(elem_types[inp.type.tensor_type.elem_type]).reshape(1,shape[-1])
            ort_inputs[inp.name] = np.repeat(ort_inputs[inp.name],repeats=max(dims_map['batch'],dims_map['batch_size']),axis=0).reshape(shape)
        elif inp.name=='attention_mask' or inp.name=='token_type_ids':
            ort_inputs[inp.name] = np.ones(shape).astype(elem_types[inp.type.tensor_type.elem_type])
        elif len(shape)==0:
            ort_inputs[inp.name] = np.random.rand((1)).astype(elem_types[inp.type.tensor_type.elem_type])
        else:
            ort_inputs[inp.name] = np.random.rand(*shape).astype(elem_types[inp.type.tensor_type.elem_type])
    in_dir = 'inputFiles/'
    os.makedirs(in_dir,exist_ok=True)
    for inp in ort_inputs:
        if os.path.exists(in_dir+inp+'.raw'):
            ort_inputs[inp] = np.fromfile(in_dir+inp+'.raw',ort_inputs[inp].dtype).reshape(ort_inputs[inp].shape)
        else:
            ort_inputs[inp].tofile(in_dir+inp+'.raw')
        print(inp, ort_inputs[inp].shape, ort_inputs[inp].dtype)
    session = onnxruntime.InferenceSession(onnx_model_path)
    ort_outputs = session.run(None, ort_inputs)
    out_dir = 'onnxRTOutputs/'
    os.makedirs(out_dir,exist_ok=True)
    for out,node in zip(ort_outputs,model.graph.output):
        out.tofile(out_dir+node.name+'.raw')
        print(node.name, out.shape, out.dtype)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
		description='save sameple inputs and reference onnxrt outputs as raw files')
    parser.add_argument("--model-path",
                        dest='model_path',
                        required=True,
                        help="onnx model path")
    parser.add_argument("--batch-size",
                        dest='batch_size',
                        required=True,
                        help="batch_size")
    parser.add_argument("--sequence",
                        dest='sequence',
                        required=False,
                        help="sequence")
    args = parser.parse_args()
    if args.sequence is not None:
        generate_inputs(args.model_path, args.batch_size, args.sequence)
    else:
        generate_inputs(args.model_path, args.batch_size)


