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
# in whole or in part, nor its contents revealed in any manner to other

import collections
import json
import math
import os
import random
import re
import shutil
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering
import tensorflow as tf
from argparse import ArgumentParser

class GenerateModels(object):
    """
    This class will be used to generate the BERT ONNX, Pytorch and TensorFlow Models
    """
    def __init__(self, config, tf_path):
        self.model = BertForQuestionAnswering(config)
        self.model.classifier = self.model.qa_outputs

        # This part is copied from HuggingFace Transformers with a fix to bypass an error
        init_vars = tf.train.list_variables(tf_path)
        names = []
        arrays = []
        for name, shape in init_vars:
            array = tf.train.load_variable(tf_path, name)
            names.append(name)
            arrays.append(array)

        for name, array in zip(names, arrays):
            name = name.split("/")
            # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
            # which are not required for using pretrained model
            if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
                continue
            pointer = self.model
            for m_name in name:
                if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                    scope_names = re.split(r"_(\d+)", m_name)
                else:
                    scope_names = [m_name]
                if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                    pointer = getattr(pointer, "bias")
                elif scope_names[0] == "output_weights":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "squad":
                    pointer = getattr(pointer, "classifier") # This line is causing the issue
                else:
                    try:
                        pointer = getattr(pointer, scope_names[0])
                    except AttributeError:
                        continue
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]
            if m_name[-11:] == "_embeddings":
                pointer = getattr(pointer, "weight")
            elif m_name == "kernel":
                array = np.transpose(array)
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            pointer.data = torch.from_numpy(array)

        self.model.qa_outputs = self.model.classifier
        del self.model.classifier
        torch.save(self.model.state_dict(), "./mlCommonsBertFiles/model.pytorch")

    def saveModels(self, args):
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = self.model.eval()
        dummy_input = torch.ones((1, 384), dtype=torch.int64)
        
        if args.save_onnx_variable_sl:
            print("\n========================== Generating ONNX Opset 11 Bert Model ======================\n")
            torch.onnx.export(
                model,
                (dummy_input, dummy_input, dummy_input),
                "./generatedModels/ONNX/BERT_MLCommons_Flexible_BS_SL.onnx",
                verbose=True,
                input_names = ["input_ids", "input_mask", "segment_ids"],
                output_names = ["output_start_logits", "output_end_logits"],
                opset_version=11,
                dynamic_axes=({"input_ids": {0: "batch_size", 1: "seq_length"}, "input_mask": {0: "batch_size", 1: "seq_length"}, 
                               "segment_ids": {0: "batch_size", 1: "seq_length"}, "output_start_logits": {0: "batch_size", 1: "seq_length"}, 
                               "output_end_logits": {0: "batch_size", 1: "seq_length"}})
            )
        if args.save_onnx_fixed_sl:
            print("\n========================== Generating ONNX Opset 11 Bert Model ======================\n")
            torch.onnx.export(
                model,
                (dummy_input, dummy_input, dummy_input),
                "./generatedModels/ONNX/BERT_MLCommons_Fixed_BS_SL.onnx",
                verbose=True,
                input_names = ["input_ids", "input_mask", "segment_ids"],
                output_names = ["output_start_logits", "output_end_logits"],
                opset_version=11
            )
        if args.save_torch_script:
            with torch.no_grad():
                print("\n==================== Generating Pytorch TorchScript Model ========================\n")
                traced = torch.jit.trace(model, (dummy_input, dummy_input, dummy_input), optimize=False)
                traced.save("./generatedModels/Pytorch/BERT_MLCommons_1_384_Pytorch.pt")
                print("Torch Trace file generated")

def parse_args():
    parser = ArgumentParser(description="BertLarge Model details")
    parser.add_argument("--config",
                        type=str,
                        required=False,
                        default="./mlCommonsBertFiles/bert_config.json",
                        help="Provide the Bert Config file")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=False,
                        default="./mlCommonsBertFiles/model.ckpt-5474",
                        help="Provide the TF Checkpoint file")
    parser.add_argument("--save-onnx-variable-sl",
                        action='store_true',
                        dest='save_onnx_variable_sl',
                        default=False,
                        help="Saves the ONNX Model")
    parser.add_argument("--save-onnx-fixed-sl",
                        action='store_true',
                        dest='save_onnx_fixed_sl',
                        default=False,
                        help="Saves the ONNX Model")
    parser.add_argument("--save-torch-script",
                        action='store_true',
                        dest='save_torch_script',
                        default=False,
                        help="Saves the torchScript Model")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        config_json = json.load(f)
    config = BertConfig(
        attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
        hidden_act=config_json["hidden_act"],
        hidden_dropout_prob=config_json["hidden_dropout_prob"],
        hidden_size=config_json["hidden_size"],
        initializer_range=config_json["initializer_range"],
        intermediate_size=config_json["intermediate_size"],
        max_position_embeddings=config_json["max_position_embeddings"],
        num_attention_heads=config_json["num_attention_heads"],
        num_hidden_layers=config_json["num_hidden_layers"],
        type_vocab_size=config_json["type_vocab_size"],
        vocab_size=config_json["vocab_size"])

    app = GenerateModels(config, args.checkpoint)
    app.saveModels(args)

if __name__ == "__main__":
    main()
