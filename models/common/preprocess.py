####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import numpy as np

from typing import Tuple
from PIL import Image

def load_image_nchw_rgb(image_path: str, input_size: Tuple[int, int] = (640, 640), batch: int = None, normalize: bool = False) -> np.ndarray:
    """
    Load image, resize to input_size, convert to NCHW float32 [0,1].
    Returns:
        img: [1, 3, H, W]
        orig_size: (orig_w, orig_h)
        resized_size: (new_w, new_h)
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    new_w, new_h = input_size
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    arr = np.array(img_resized).astype(np.float32)  # HWC
    if normalize:
        arr /= 255.0 # [0,1]
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    if batch:
        arr = np.expand_dims(arr, 0)        # NCHW

    return np.ascontiguousarray(arr)
