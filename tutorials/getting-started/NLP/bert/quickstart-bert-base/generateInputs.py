##############################################################################
#
#Copyright (c) 2022 Qualcomm Technologies, Inc.
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
from argparse import ArgumentParser

import numpy as np
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer

model_name = "madlag/bert-base-uncased-squadv1-x2.01-f89.2-d30-hybrid-rewind-opt-v1"
config = BertConfig.from_pretrained(model_name, return_dict=False)
tokenizer = BertTokenizer.from_pretrained(model_name)
question = "How many trees are there in amazon forest?"
questionContext = "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain Amazonas in their names. The Amazon represents over half of the planets remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."


def get_single_input(seq_length):
    encodings = tokenizer.encode_plus(question, questionContext, padding="max_length")
    inputIds = np.array([encodings["input_ids"]])[:, :seq_length].astype(np.int64)
    attentionMask = np.array([encodings["attention_mask"]])[:, :seq_length].astype(
        np.int64
    )
    tokenTypeIds = np.zeros((1, inputIds.shape[1])).astype(np.int64)
    return inputIds, attentionMask, tokenTypeIds


def generate_batched_inputs(batch_size, seq_length):

    inputIds = []
    attentionMasks = []
    tokenTypeIds = []

    for i in range(batch_size):
        inputId, attentionMask, tokenTypeId = get_single_input(seq_length)
        inputIds.append(inputId)
        attentionMasks.append(attentionMask)
        tokenTypeIds.append(tokenTypeId)

    inputIds = np.array(inputIds).reshape(batch_size, seq_length)
    attentionMasks = np.array(attentionMasks).reshape(batch_size, seq_length)
    tokenTypeIds = np.array(tokenTypeIds).reshape(batch_size, seq_length)

    input_file_dir="inputFiles"
    
    os.makedirs(input_file_dir, exist_ok=True)
    inputIds.tofile(input_file_dir+f"/input_ids_sl{seq_length}_bs{batch_size}.raw")
    attentionMasks.tofile(input_file_dir+f"/input_mask_sl{seq_length}_bs{batch_size}.raw")
    tokenTypeIds.tofile(input_file_dir+f"/segment_ids_sl{seq_length}_bs{batch_size}.raw")
    
    file1 = open("input_list.txt","w+")
    
    file1.write(f"{input_file_dir}/input_ids_sl{seq_length}_bs{batch_size}.raw,")
    file1.write(f"{input_file_dir}/input_mask_sl{seq_length}_bs{batch_size}.raw,")
    file1.write(f"{input_file_dir}/segment_ids_sl{seq_length}_bs{batch_size}.raw")
    file1.close()


def parse_args():
    parser = ArgumentParser(description="BertBase Input Processing details")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        required=False,
        help="Generate QA input files",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help="Generate QA input files",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    generate_batched_inputs(args.batch_size, args.seq_len)


if __name__ == "__main__":
    main()
