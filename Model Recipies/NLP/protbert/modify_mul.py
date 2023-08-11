import onnx
import numpy as np
from qaic_pytools.qmodel.backend.onnx.model import QOnnxModel
from qaic_pytools.qmodel.backend.onnx.utils import get_model_node_mappings, get_initializer_value

model_path = "generatedModels/ProtBert.onnx/model.onnx"
qmodel = QOnnxModel(model_path)
const_nodes = qmodel.get_nodes_by_op_type("Constant")
mul_nodes = qmodel.get_nodes_by_op_type("Mul")
_, get_initializer_by_name, _, _, _ = get_model_node_mappings(qmodel)
for node in mul_nodes:
    for node_ip in node.input:
        if node_ip in get_initializer_by_name:
            init = get_initializer_by_name[node_ip]
            init_value = get_initializer_value(init)
            if len(init_value.flatten()) == 1 and init_value < -1e-6:
                new_init_value = np.array(-10000.0, dtype=np.float32)
                new_initializer = onnx.numpy_helper.from_array(new_init_value, name=init.name)
                init.CopyFrom(new_initializer)
for node in const_nodes:
    init = node.attribute[0].t
    init_value = get_initializer_value(init)
    if len(init_value.flatten()) == 1 and init_value < -1e+6:
        new_init_value = np.array(-10000.0, dtype=np.float32)
        new_initializer = onnx.numpy_helper.from_array(new_init_value, name=init.name)
        node.attribute[0].t.CopyFrom(new_initializer)

qmodel.summarize_model()
qmodel.save_model("generatedModels/ProtBert.onnx/model_modified.onnx")