####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from importlib import import_module

# Thin wrapper that re-exports the original implementation from
# models.vision.segmentation.mask_rcnn.postprocess
_impl = import_module("models.vision.segmentation.mask_rcnn.model")

for name in dir(_impl):
    if not name.startswith("_"):
        globals()[name] = getattr(_impl, name)
