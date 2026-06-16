####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from importlib import import_module

# Thin wrapper that re-exports the original implementation from
# models.common.postprocess
_impl = import_module("models.common.postprocess")

for name in dir(_impl):
    if not name.startswith("_"):
        globals()[name] = getattr(_impl, name)
