####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import numpy as np
from typing import Tuple
from qaic_models.common.preprocess import load_image_nchw_rgb

class MaskRCNNImage:
    """Utility for loading and preprocessing images for Mask R-CNN.
    """

    def load(image_path: str, size: Tuple[int, int] = (1216, 800), batch: int = None) -> np.ndarray:
        return load_image_nchw_rgb(image_path, size, batch, normalize=False)