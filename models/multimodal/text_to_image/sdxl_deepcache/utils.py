import os
import onnx
from onnx import numpy_helper


# executes the command and writes it down in the command.txt. The first time, mode is 'w', then 'a' (append)
def execute(cmd_elements, write_to_file, mode):
    cmd_str = ' '.join(str(x) for x in cmd_elements)
    redirect = f" 2>&1 | ts > {write_to_file}"
    cmd_str += redirect
    print(f"Executing: {cmd_str}")
    os.system(cmd_str)
    with open(write_to_file, mode) as file:
        file.write(cmd_str + "\n\n")


def scale_conv(model, conv_name, scale_factor):
    cnodes = [x for x in model.graph.node if x.name == conv_name]
    assert len(cnodes) == 1, f"Node '{conv_name}' not found"
    x, w, b = cnodes[0].input
    wi, bi = "", ""
    for i, init in enumerate(model.graph.initializer):
        if init.name == w:
            wi = i
        elif init.name == b:
            bi = i
        if wi != "" and bi != "":
            break
    else:
        raise ValueError(f"Cannot find indices of weight: {w} and bias: {b}")
    ww = numpy_helper.to_array(model.graph.initializer[wi])
    bb = numpy_helper.to_array(model.graph.initializer[bi])
    model.graph.initializer[wi].raw_data = (ww / scale_factor).tobytes()
    model.graph.initializer[bi].raw_data = (bb / scale_factor).tobytes()
