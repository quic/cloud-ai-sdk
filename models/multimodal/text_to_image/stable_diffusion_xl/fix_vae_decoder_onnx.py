####################################################################################################
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
import onnx
from onnx import numpy_helper

def scale_conv(model, conv_name, scale_factor):
    cnodes = [x for x in model.graph.node if x.name == conv_name]
    assert len(cnodes) == 1, f"Node '{conv_name}' not found"
    x, w, b = cnodes[0].input
    wi, bi = "", ""
    for i, init in enumerate(model.graph.initializer):
        if init.name == w:
            wi = i
        elif init.name == b:
            bi = i
        if wi != "" and bi != "":
            break
    else:
        raise ValueError(f"Cannot find indices of weight: {w} and bias: {b}")
    ww = numpy_helper.to_array(model.graph.initializer[wi])
    bb = numpy_helper.to_array(model.graph.initializer[bi])
    model.graph.initializer[wi].raw_data = (ww / scale_factor).tobytes()
    model.graph.initializer[bi].raw_data = (bb / scale_factor).tobytes()


def main(model_path, scaling_factor):
    model = onnx.load(model_path)
    scale_conv(model, "/decoder/up_blocks.2/upsamplers.0/conv/Conv", scaling_factor)
    scale_conv(model, "/decoder/up_blocks.3/resnets.0/conv2/Conv", scaling_factor)
    # scale_conv(model, "/decoder/up_blocks.3/resnets.0/conv_shortcut/Conv", scaling_factor)
    scale_conv(model, "/decoder/up_blocks.3/resnets.1/conv2/Conv", scaling_factor)
    scale_conv(model, "/decoder/up_blocks.3/resnets.2/conv2/Conv", scaling_factor)
    output_path = model_path[:-5] + f"_fixed_{scaling_factor}.onnx"
    onnx.save(model, output_path)


if __name__ == "__main__":
    import argparse
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--model-path",
        default="stabilityai/stable-diffusion-xl-base-1.0/vae_decoder/model.onnx",
        help="Model path to fix",
    )
    argp.add_argument("--scaling-factor", default=128, type=int, help="Scaling factor")
    args = argp.parse_args()
    main(**vars(args))
