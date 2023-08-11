"""
Qualcomm Technologies, Inc. Proprietary
(c) 2023 Qualcomm Technologies, Inc. All rights reserved.
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

sys.path.insert(0, os.path.join(os.getcwd(), "yolov7"))
from utils.datasets import letterbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="File to Generate raw files")
    parser.add_argument("--img_path",
                        required=True,
                        default="./images/horses.jpg",
                        type=str,
                        help="Path to image file")
    parser.add_argument("--h_w",
                        type=int,
                        nargs='+',
                        default=[640, 640],
                        help="Give Input resolutions, height, width")
    args = parser.parse_args()

    im0s = cv2.imread(args.img_path)
    img = letterbox(im0s, args.h_w, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(
        np.float32)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    img.tofile(f"inputFiles/input.raw")
