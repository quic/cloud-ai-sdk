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
import numpy as np
from argparse import ArgumentParser

torch.manual_seed(10)


def genRandomData(size,bs):
    input_list_file = open(f"input_list.txt","w+")

    inputIds = torch.randint(0, 10000, (bs,1, size))
    attentionMask = torch.ones((bs,1, size))

    mask = inputIds.ne(1).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
    positionIds =incremental_indices.long() + 1
    
    raw_dir="inputFiles"
    os.makedirs(raw_dir,exist_ok=True)
    
    inputIds.numpy().astype(np.int64).tofile(f"{raw_dir}/input_id.raw")
    attentionMask.numpy().astype(np.int64).tofile(f"{raw_dir}/attention_mask.raw")
    positionIds.numpy().astype(np.int64).tofile(f"{raw_dir}/position_id.raw")
    
    input_list_file.write(f"{raw_dir}/input_id.raw,{raw_dir}/attention_mask_sl.raw,{raw_dir}/position_id_sl.raw \n")
    input_list_file.close()

def parse_args():
    parser = ArgumentParser(description="Generate sample data")
    parser.add_argument(
        "--sl",
        type=int,
        default=128,
        help="Sequence length for the input",
    )
    parser.add_argument(
            "--bs",
            type=int,
            default=1,
            help=" batch size ",)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    genRandomData(args.sl,args.bs)
