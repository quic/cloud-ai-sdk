'''
Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import os
import yaml
import inspect

def generate_bin(onnx_filename, yaml_filename):
    """
    Generate compiled binary for QAIC

    Args:
        onnx_path : path to onnx file.
        yaml_path : path to yaml file which has compile time arguments.

    Returns:
        qpc_path : path to qpc (compiled binary)
    """
    caller_path = inspect.stack()[1].filename #os.path.dirname(os.path.realpath
    onnx_path = os.path.join(os.path.dirname(caller_path), onnx_filename)
    yaml_path = os.path.join(os.path.dirname(caller_path), yaml_filename)

    filename, extension = os.path.splitext(onnx_filename)
    onnx_folder = os.path.dirname(onnx_path)
    qpc_bin = os.path.join(os.path.dirname(caller_path), filename+'_qpc')
    with open(yaml_path, "r") as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)

    if os.path.isdir(qpc_bin):
        print(f'INFO: Removing existing QPC {qpc_bin}')
        cmd = f'sudo rm -fr {qpc_bin}'
        os.system(cmd)
        print(f'INFO: Existing QPC {qpc_bin} is removed')

    # create the command string from the yaml arguments. 
    cmd_list = [f'/opt/qti-aic/exec/qaic-exec -m={onnx_path} -aic-hw -aic-hw-version={2.0}']

    # ignore the following arguments:
    ignore = ['num-activations', 'set-size']
    replace_dict = {'aic_num_cores':'aic-num-cores'}

    for arg, value in yaml_data.items():
        arg = arg.replace('_','-')
        if arg in ignore:
            continue
        if isinstance(value, bool):
            if value:# include the argument only if true; for example -convert-to-fp16
                cmd_list.append(f'-{arg}') 
        elif isinstance(value, dict):
            for subarg, subval in value.items():
                cmd_list.append(f'-{arg}={subarg},{subval}')
        else:
            cmd_list.append(f'-{arg}={value}')

    cmd_list.append(f'-aic-binary-dir={qpc_bin}')

    cmd = ' '.join(cmd_list)
    print(f'INFO: Running the compile cmd: {cmd}')
    os.system(cmd)
    
    return qpc_bin
