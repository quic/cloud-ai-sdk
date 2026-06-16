####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from importlib import import_module
from typing import Any

# Thin wrapper that re-exports the original implementation from
# models.vision.segmentation.mask_rcnn.demo and provides a main() entry
# point so it can be invoked as a module:
#   python -m qaic_models.vision.segmentation.mask_rcnn.demo
_impl = import_module("models.vision.segmentation.mask_rcnn.demo")

for name in dir(_impl):
    if not name.startswith("_"):
        globals()[name] = getattr(_impl, name)


def main() -> None:
    """Delegate to the original demo.main() if present."""
    impl_main: Any = getattr(_impl, "main", None)
    if callable(impl_main):
        impl_main()


if __name__ == "__main__":
    main()
