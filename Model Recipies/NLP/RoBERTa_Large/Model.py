"""
##############################################################################
#
#Copyright (c) 2021 Qualcomm Technologies, Inc.
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
"""
import os
import torch
from argparse import ArgumentParser
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig

torch.manual_seed(10)

question = "How many trees are there in amazon forest?"
questionContext = "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain Amazonas in their names. The Amazon represents over half of the planets remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."

def get_example_input(tokenizer, config):
    encodings = tokenizer.encode_plus(question, questionContext)
    inputIds = encodings["input_ids"]
    attentionMask = encodings["attention_mask"]

    mask = torch.tensor([inputIds]).ne(config.pad_token_id).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
    positionIds = incremental_indices.long() + config.pad_token_id

    return inputIds, attentionMask, positionIds

def generateAndRunModel(args):
    """
    This API will generate the ONNX and Pytorch models and executes them using their
    standalone inference engines
    """
    config = AutoConfig.from_pretrained(args.model_card, return_dict=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_card)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_card, config=config)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # dummy input shape
    inputIds, attentionMask, positionIds = get_example_input(tokenizer, config)

    if args.exclude_cumsum_op:
        inputList = [torch.tensor([inputIds]), torch.tensor([attentionMask]), None, positionIds]
    else:
        inputList = [torch.tensor([inputIds]), torch.tensor([attentionMask])]

    if args.save_input:
        inputList[0].numpy().astype(np.int64).tofile(f"./inputFiles/input_id_{len(inputIds)}.raw")
        inputList[1].numpy().astype(np.int64).tofile(f"./inputFiles/attention_mask_{len(attentionMask)}.raw")
        if args.exclude_cumsum_op:
            inputList[3].numpy().astype(np.int64).tofile(f"./inputFiles/position_id_{positionIds.shape[0]}.raw")

    dynamic_dims = {0 : 'batch', 1: "sequence"}
    dynamic_axes = {
                "input_ids": dynamic_dims,  # variable length axes
                "attention_mask": dynamic_dims,  # variable length axes
                "start_scores": dynamic_dims,
                "end_scores": dynamic_dims,
            }
    input_names = ["input_ids", "attention_mask"]

    if args.exclude_cumsum_op:
        dynamic_axes["position_ids"] = dynamic_dims
        input_names += ["position_ids"]

    torch.onnx.export(
        model,
        args=tuple(inputList),
        f=args.output_path,
        verbose=False,
        input_names=input_names,
        output_names=["start_scores", "end_scores"],
        dynamic_axes=dynamic_axes,
        opset_version=11,
    )
    print("ONNX Model is being generated successfully for opset Version 11")

    if args.run_onnx_standalone_inference:
        print("\n", "="*40, "ONNX Runtime Standalone", "="*40, "\n")
        import onnxruntime

        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        ort_session = onnxruntime.InferenceSession(
            args.output_path, sess_opts
        )

        input_dict = {"input_ids": [inputIds], "attention_mask": [attentionMask]}
        if args.exclude_cumsum_op:
            input_dict["position_ids"] = [positionIds[0].cpu().numpy()]
        ort_outs = ort_session.run(None, input_dict)
        ort_outs[0] = torch.from_numpy(ort_outs[0])
        ort_outs[1] = torch.from_numpy(ort_outs[1])

        tokens = inputIds[torch.argmax(ort_outs[0]) : torch.argmax(ort_outs[1]) + 1]
        answerTokens = tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True)
        answer = tokenizer.convert_tokens_to_string(answerTokens)
        print("ONNX Question: ", question)
        print("ONNX Answer: ", answer)


def parse_args():
    parser = ArgumentParser(description="RoBerta Model details")
    parser.add_argument(
        "--model-card",
        dest="model_card",
        default="roberta-base",
        help="Model card for the roberta-model. Default: roberta-base",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        required=True,
        help="Output path of the model",
    )
    parser.add_argument(
        "--exclude-cumsum-op",
        action="store_true",
        dest="exclude_cumsum_op",
        default=False,
        help="Exclude cumsum op in the ONNX model",
    )
    parser.add_argument(
        "--save-input",
        action="store_true",
        dest="save_input",
        default=False,
        help="Save the input to a raw file",
    )
    parser.add_argument(
        "--run-onnx-standalone-inference",
        action="store_true",
        dest="run_onnx_standalone_inference",
        default=False,
        help="Run the Standalone ONNX Inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generateAndRunModel(args)


if __name__ == "__main__":
    main()
