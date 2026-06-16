####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from importlib import import_module

# Delegate to the original implementation under models.vision.segmentation.mask_rcnn
_impl = import_module("models.vision.segmentation.mask_rcnn")

# Re-export all public attributes from the underlying module
for name in dir(_impl):
    if not name.startswith("_"):
        globals()[name] = getattr(_impl, name)
