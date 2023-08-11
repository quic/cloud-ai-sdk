"""
##############################################################################
#
#Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
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
torch.manual_seed(10)
from argparse import ArgumentParser
from transformers import DebertaV2Tokenizer, DebertaV2Config, DebertaV2ForSequenceClassification
from pdb import set_trace as bp
model_name = 'microsoft/deberta-v3-xsmall'
config = DebertaV2Config.from_pretrained(model_name, return_dict=False)
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
input_sequence = "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain Amazonas in their names. The Amazon represents over half of the planets remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."
def get_example_input(compressed=True):
    encodings = tokenizer.encode_plus(input_sequence, max_length=256, padding='max_length')
    inputIds = torch.tensor([encodings["input_ids"]])
    attentionMask = torch.tensor([encodings["attention_mask"]])
    if compressed:
        inputLens = torch.sum(attentionMask, axis=1)
        return inputIds, inputLens
    return inputIds, attentionMask
def generateAndRunModel(args):
    """
    This API will generate the ONNX and Pytorch models and executes them using their
    standalone inference engines
    """
    model = DebertaV2ForSequenceClassification.from_pretrained(model_name, config=config)
    model.eval()
    # dummy input shape
    inputIds, attentionMask = get_example_input(args.compressed)
    inputList = [inputIds, attentionMask]
    if args.save_onnx:
        onnx_model_path = "./generatedModels/Onnx/"+model_name+"-classification.onnx"
        if args.compressed:
            onnx_model_path = onnx_model_path[:-5] + '_compressedmask.onnx'
        if not os.path.isdir(os.path.dirname(onnx_model_path)):
            os.makedirs(os.path.dirname(onnx_model_path))
        torch.onnx.export(
            model,
            args=tuple(inputList),
            f=onnx_model_path,
            verbose=False,
            input_names=["input_ids", "attention_mask"],
            output_names=["probs"],
            dynamic_axes={
                "input_ids": {0 : 'batch', 1: "sequence"},  # variable length axes
                "attention_mask": {0 : 'batch', 1: "sequence"}},
            opset_version=13,
        )
        print("ONNX Model is being generated successfully for opset Version 13")
    if args.save_raw_files:
        import numpy as np
        inputIdsOnnx = inputIds.cpu().numpy().astype(np.float32)
        attentionMaskOnnx = attentionMask.cpu().numpy().astype(np.float32)
        
        if args.bs > 1:
            inputIdsOnnx = np.array([inputIdsOnnx]*args.bs)
            attentionMaskOnnx = np.array([attentionMaskOnnx]*args.bs)
            
        inputs_dir = 'inputFiles/'
        if not os.path.isdir(inputs_dir):
            os.makedirs(inputs_dir)
        inputIdsOnnx.tofile(inputs_dir+f"/input_ids_bs{args.bs}.raw")
        attentionMaskOnnx.tofile(inputs_dir+f"/attention_mask_bs{args.bs}.raw")
        
        file1 = open(f"inputlist_bs{args.bs}.txt","w+")
        
        file1.write(f"{inputs_dir}/input_ids_bs{args.bs}.raw,")
        file1.write(f"{inputs_dir}/attention_mask_bs{args.bs}.raw")        
        file1.close()
        
        
def parse_args():
    parser = ArgumentParser(description="RoBerta Model details")
    parser.add_argument(
        "--save_onnx",
        action="store_true",
        dest="save_onnx",
        default=False,
        help="Save the onnx Graph from Pytorch model",
    )
    parser.add_argument(
        "--compressed_mask_model",
        action="store_true",
        dest="compressed",
        default=False,
        help="Save the Compressed mask onnx model",
    )
    parser.add_argument(
        "--save_raw_files",
        action="store_true",
        dest="save_raw_files",
        default=False,
        help="Save Raw input files and Onnxrt output files",
    )
    parser.add_argument(
        "--bs",
        default=1,
        type=int,
        help="batch size",
    )
    
    
    return parser.parse_args()
def main():
    args = parse_args()
    generateAndRunModel(args)
if __name__ == "__main__":
    main()
