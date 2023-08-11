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
# LICENCE: https://github.com/huggingface/transformers/blob/main/LICENSE
#          Apache License
#    Version 2.0, January 2004
# http://www.apache.org/licenses/
# SOURCE: https://huggingface.co/mrm8488/electra-small-finetuned-squadv2

##############################################################################

import re
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from argparse import ArgumentParser

torch.manual_seed(10)
sequence_example = "A E T C Z A O"
sequence_example = re.sub(r"[UZOB]", "X", sequence_example)

# public model
def generateInputs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_card)
    # run tokenizer on inputs
    tokenizer_kwargs = {'truncation': True, 'max_length': 128, 'padding': 'max_length'}
    model_inputs = tokenizer(sequence_example, return_tensors='np')
    # mask_token_index = torch.where(model_inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    model_inputs['input_ids'].tofile(f"{args.output_path}/input_ids.raw")
    model_inputs["attention_mask"].tofile(f"{args.output_path}/attention_mask.raw")
    model_inputs['token_type_ids'].tofile(f"{args.output_path}/token_type_ids.raw")

    print(model_inputs["attention_mask"])

    print(np.sum(model_inputs["attention_mask"]))
    modified_attention = np.sum(model_inputs["attention_mask"]).reshape(1, 1)
    modified_attention.tofile(f"{args.output_path}/attention_masked.raw")
    print(modified_attention, modified_attention.shape)

def parse_args():
    parser = ArgumentParser(description="BioBert Model details")
    parser.add_argument(
        "--model-card",
        dest="model_card",
        default='Rostlab/prot_bert',
        help="Model card for the sentence-transformers model."
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        required=True,
        help="Output path of the Inputs",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    generateInputs(args)


if __name__=="__main__":
    main()