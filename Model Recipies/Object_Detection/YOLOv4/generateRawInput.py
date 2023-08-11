"""
Qualcomm Technologies, Inc. Proprietary
(c) 2020 Qualcomm Technologies, Inc. All rights reserved.
All data and information contained in or disclosed by this document are
confidential and proprietary information of Qualcomm Technologies, Inc., and
all rights therein are expressly reserved. By accepting this material, the
recipient agrees that this material and the information contained therein
are held in confidence and in trust and will not be used, copied, reproduced
in whole or in part, nor its contents revealed in any manner to others
without the express written permission of Qualcomm Technologies, Inc.
"""
import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="File to Generate raw files")
    parser.add_argument("--image_path",
                        required=True,
                        default="./images/zidane.jpg",
                        type=str,
                        help="Path to image file")
    parser.add_argument("--image_height_width",
                        required=True,
                        type=int,
                        nargs='+',
                        help="Give Input resolutions, height, width")
    args = parser.parse_args()
    image_src = Image.open(args.image_path)
    resized = letterbox_image(image_src, tuple(args.image_height_width))
    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in.tofile("input_" + str(args.image_height_width[1]) + "_" + str(args.image_height_width[0]) + ".raw")
    print("Generate Raw File Shape is: ", img_in.shape)