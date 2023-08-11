##############################################################################
#
#Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#All Rights Reserved.
#Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#All data and information contained in or disclosed by this document are
#confidential and proprietary information of Qualcomm Technologies, Inc., and
#all rights therein are expressly reserved. By accepting this material, the
#recipient agrees that this material and the information contained therein
#are held in confidence and in trust and will not be used, copied, reproduced
#in whole or in part, nor its contents revealed in any manner to others
#without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################

import os
import torch
import urllib
from argparse import ArgumentParser
torch.manual_seed(10)
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from transformers.pipelines import Pipeline, pipeline
from transformers.data.processors.squad import SquadExample
from typing import Dict, List, Optional, Tuple
from transformers.tokenization_utils import BatchEncoding
from transformers.file_utils import ModelOutput, is_tf_available, is_torch_available
from pathlib import Path

def openfile(pathname):
    dirpath = os.path.dirname(pathname)
    os.makedirs(dirpath, exist_ok=True)
    return pathname

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

QA_inp={
    'question': 'How many parameters does Bert large have?',
    'context': 'Bert is a deep learning model that has given state-of-the-art results on a wide variety of natural language processing tasks. It stands for Bidirectional Encoder Representations for Transformers. It has been pre-trained on Wikipedia and BooksCorpus and requires task-specific fine-tuning. multi-layer bidirectional Transformer encoder. Bert large is really big. It has 24 layers, for a total of 340M parameters.Altogether it is 1.34 GB so expect it to take a couple minutes to download to your Colab instance.'
}

QA_inp = dotdict(QA_inp)

def parse_args():
    parser = ArgumentParser(description="Model Generation Help")
    parser.add_argument(
        "--onnx-output",
        type=str,
        dest="onnx_output",
        default="./model.onnx",
        help="Save the onnx Graph from Pytorch model to this file",
    )
    parser.add_argument(
        "--model-card",
        type=str,
        required=False,
        dest="model_card",
        default='https://huggingface.co/distilroberta-base',
        help="Give HF model card link, e.g https://huggingface.co/distilroberta-base",
    )
    return parser.parse_args()

def get_model_name(link):
    if link is not None:
        if "https://huggingface.co" in link :
            import requests
            response = requests.get(link)
            if response.status_code == 200:
                parsed = urllib.parse.urlparse(link)
                return parsed.path[1:]
            else:
                print('Invalid Hugging face link')

def ensure_valid_input(model, tokens, input_names):
    model_args_name = model.forward.__code__.co_varnames
    model_args, ordered_input_names = [], []
    for arg_name in model_args_name[1:]:  # start at index 1 to skip "self" argument
        #print(arg_name)
        if arg_name in input_names:
            ordered_input_names.append(arg_name)
            model_args.append(tokens[arg_name])
        else:
            continue
    return ordered_input_names, model_args

def infer_shapes(nlp: Pipeline, framework: str, example:SquadExample) -> Tuple[List[str], List[str], Dict, BatchEncoding]:
    def build_shape_dict(name: str, tensor, is_input: bool, seq_len: int):
        if isinstance(tensor, (tuple, list)):
            return [build_shape_dict(name, t, is_input, seq_len) for t in tensor]
        else:
            # Let's assume batch is the first axis with only 1 element (~~ might not be always true ...)
            axes = {[axis for axis, numel in enumerate(tensor.shape) if numel == 1][0]: "batch"}
            #print(axes)
            if is_input:
                if len(tensor.shape) == 2:
                    axes[1] = "sequence"
                else:
                    raise ValueError(f"Unable to infer tensor axes ({len(tensor.shape)})")
            else:
                seq_axes = [dim for dim, shape in enumerate(tensor.shape) if shape == seq_len]
                axes.update({dim: "sequence" for dim in seq_axes})
        #print(f"Found {'input' if is_input else 'output'} {name} with shape: {axes}")
        return axes
    tokens = nlp.tokenizer(example.question_text, example.context_text, return_tensors=framework)
    seq_len = tokens.input_ids.shape[-1]
    with torch.no_grad():
        outputs = nlp.model(**tokens) if framework == "pt" else nlp.model(tokens)
    if isinstance(outputs, ModelOutput):
        outputs = outputs.to_tuple()
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
    # Generate input names & axes
    input_vars = list(tokens.keys())
    input_dynamic_axes = {k: build_shape_dict(k, v, True, seq_len) for k, v in tokens.items()}
    # flatten potentially grouped outputs (past for gpt2, attentions)
    outputs_flat = []
    for output in outputs:
        if isinstance(output, (tuple, list)):
            outputs_flat.extend(output)
        else:
            outputs_flat.append(output)
    extra_inputs = nlp.preprocess(example)
    final_result = nlp.postprocess({"starts" : outputs_flat[0], "ends":outputs_flat[1], "features":extra_inputs['features'], "example":extra_inputs['example']})
    # Generate output names & axes
    # output_names = [f"output_{i}" for i in range(len(outputs_flat))]
    output_names = ["start_logits", "end_logits"]
    output_dynamic_axes = {k: build_shape_dict(k, v, False, seq_len) for k, v in zip(output_names, outputs_flat)}
    # Create the aggregated axes representation
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    return input_vars, output_names, dynamic_axes, tokens

def enable_weight_sharing(onnx_model_filename):
    from onnxruntime.transformers.onnx_model import OnnxModel
    import onnx
    import numpy as np
    model=onnx.load(onnx_model_filename)
    onnx_model=OnnxModel(model)
    count = len(model.graph.initializer)
    same = [-1] * count
    for i in range(count - 1):
        if same[i] >= 0:
            continue
        for j in range(i+1, count):
            if model.graph.initializer[i].dims == model.graph.initializer[j].dims:
                if model.graph.initializer[i].data_type == model.graph.initializer[j].data_type:
                # exit(1)
                    if model.graph.initializer[i].raw_data == model.graph.initializer[j].raw_data:
                        same[j] = i
    # print(same)
    for i in range(count):
        if same[i] >= 0:
            onnx_model.replace_input_of_all_nodes(model.graph.initializer[i].name, model.graph.initializer[same[i]].name)
    onnx_model.update_graph()
    onnx_model.save_model_to_file(onnx_model_filename)

def save_onnxrt_output(path : Path, input_dict, output_names):
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
    print(f"Checking ONNX model loading from: {path}")
    try:
        onnx_options = SessionOptions()
        sess = InferenceSession(path, onnx_options, providers=["CPUExecutionProvider"])
        print(f"Model {path} correctly loaded:")
    except RuntimeException as re:
        print(f"Error while loading the model")
    onnxRtOutput = sess.run(output_names,input_dict)
    for key, value in zip(output_names, onnxRtOutput):
        value.tofile(openfile(f"./onnxrt_reference_output/{key}.bin"))
        print(f"== Saving onnxrt == ./onnxrt_reference_output/{key}.bin of shape {value.shape}")
    return onnxRtOutput

def main():
    args = parse_args()
    # Generate model-card name from link
    print(f" == Generating hf model_zoo model-card from {args.model_card} ==")
    model_path_or_name = get_model_name(args.model_card)
    # Get modelclass, weights and config file from model-card
    model = AutoModelForQuestionAnswering.from_pretrained(model_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    config = AutoConfig.from_pretrained(model_path_or_name)
    print("== Exporting model with Config  ==\n", config)
    nlp_QA=pipeline('question-answering',model=model,tokenizer=tokenizer,config=config)
    example = nlp_QA.create_sample(QA_inp['question'], QA_inp['context'])
    result = nlp_QA(QA_inp)
    print("Result of complete model in pytorch: ", result)
    ####################################################################################
    #                   Generating ONNX model input raw files
    ####################################################################################
    input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp_QA, nlp_QA.framework, example)
    ordered_input_names, model_args = ensure_valid_input(nlp_QA.model, tokens, input_names)
    onnxrt_input_dict = {}
    for key,value in zip(ordered_input_names,model_args):
        data = value.numpy().astype('int64')
        # print(key, data.shape)
        onnxrt_input_dict[key] = data
        data.tofile(openfile(f"./input_dir/{key}.raw"))
        print(f"== Saving onnxrt == ./input_dir/{key}.raw of shape {value.shape}")
    ####################################################################################
    #                   Model Export to ONNX
    ####################################################################################
    print(f"== Saving onnx model to {args.onnx_output} ==")
    openfile(args.onnx_output)
    from torch.onnx import export
    export(
        nlp_QA.model,
        tuple(model_args),
        f=args.onnx_output,
        input_names=ordered_input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        use_external_data_format=False,
        enable_onnx_checker=True,
        opset_version=11,
    )
    # alternate : use onnxsimplifier to perform weight sharing
    enable_weight_sharing(args.onnx_output)
    ####################################################################################
    #                   Generating ONNX model output
    ####################################################################################
    onnxrt_output = save_onnxrt_output(args.onnx_output, onnxrt_input_dict, output_names)
    ####################################################################################
    #                   Question-Answering post-processing
    ####################################################################################
    extra_inputs = nlp_QA.preprocess(example)
    final_result = nlp_QA.postprocess({"starts" : onnxrt_output[0], "ends":onnxrt_output[1], "features":extra_inputs['features'], "example":extra_inputs['example']})
    print("Result of onnx model after post-processing : ", final_result)

if __name__ == "__main__":
    main()
