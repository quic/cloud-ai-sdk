##############################################################################
#
# Copyright (c) 2022 Qualcomm Technologies, Inc.
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
#
##############################################################################
import argparse
import os
import onnx
from onnx.tools import update_model_dims
def modify_model(model_ip_file, model_op_file):
    if not os.path.isfile(model_ip_file):
        print(f"{model_ip_file} not found. Skipping the model.")
    onnx_graph = onnx.load(model_ip_file, load_external_data=True)
    input_dims = {
        "input_ids": ["batch_size", "seq_len"],
        "attention_mask": ["batch_size", "seq_len"],
        "token_type_ids": ["batch_size", "seq_len"],
    }
    output_dims = {
        "start_logits": ["batch_size", "seq_len"],
        "end_logits": ["batch_size", "seq_len"],
    }
    final_model = update_model_dims.update_inputs_outputs_dims(
        onnx_graph, input_dims, output_dims
    )
    size_gb = final_model.ByteSize() / 1073741824.0
    if size_gb > 2.0:
        onnx.save_model(
            final_model,
            model_op_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.splitext(os.path.basename(model_op_file))[0] + ".bin",
        )
    else:
        onnx.save_model(final_model, model_op_file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modify bert base model input and output dimensions to dynamic "
        + "per AIC"
    )
    parser.add_argument(
        "--model-ip-path",
        required=True,
        dest="model_ip_file",
        help="Path for onnx Model",
    )
    parser.add_argument(
        "--model-op-path",
        required=True,
        dest="model_op_file",
        help="Path for onnx Model",
    )
    args = parser.parse_args()
    modify_model(args.model_ip_file, args.model_op_file)
