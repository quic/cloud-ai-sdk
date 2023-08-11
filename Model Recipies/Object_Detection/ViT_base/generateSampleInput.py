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
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from transformers import ViTFeatureExtractor


# python generateSampleInput.py --img_path ./inputFiles/000000039769.jpg \
# --batch_size 16 --output_path ./inputFiles/000000039769_bs16.raw

def preprocessImg(img_path, batch_size, output_path):
    if not os.path.isfile(img_path):
        print("Error: Image path not found : ", img_path)
        exit()

    single_image = Image.open(img_path)

    feature_extractor = ViTFeatureExtractor.from_pretrained(\
                                            'google/vit-base-patch16-224')
    single_img_data = feature_extractor(images=single_image, \
                                                return_tensors="np")
    preprocessed_single_img = single_img_data['pixel_values']
    # preprocessed_single_img : [1 x 3 x 224 x 224] : Numpy array
    preprocessed_batch = np.tile(preprocessed_single_img, \
                                                [batch_size, 1, 1, 1])

    preprocessed_batch = np.float32(preprocessed_batch)
    preprocessed_batch.tofile(output_path)
    print("Preprocessed tensor is stored at : ", output_path)


def parse_args():
    parser = ArgumentParser(description="Generate sample raw file based " \
                                            + "on provided image paths.")
    parser.add_argument(
        "--img_path",
        type=str,
        help="Image path")
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batchsize of the preprocessed tensor.")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path where preprocessed batch data is stored in .raw format.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocessImg(args.img_path, args.batch_size, args.output_path)
