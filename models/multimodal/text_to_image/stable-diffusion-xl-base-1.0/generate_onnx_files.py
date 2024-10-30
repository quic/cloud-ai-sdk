####################################################################################################
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
import time
import torch
import os
import shutil
from pathlib import Path

import onnx
from diffusers import StableDiffusionXLPipeline
from onnx import external_data_helper, numpy_helper
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from onnx_gen_utils import onnx_export

BATCHSIZE = 2 # to avoid collapse of batch dimension in ONNX UNet tracing
LATENT_CHANNELS = 4
LATENT_HEIGHT = 128
LATENT_WIDTH = 128
SEQUENCE_LENGTH = 77
ENCODER_HIDDEN_STATES_DIM = 2048
TEXT_EMBEDS_DIM = 1280
TIME_IDS_SIZE = 6


def export_onnx_unet(pipeline, output_path, opset):
    # UNET
    unet_path = output_path / "unet" / "model.onnx"
    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(BATCHSIZE, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH),
            torch.randint(0,20,(2,), dtype=torch.int64), # random int64
            torch.randn(BATCHSIZE, SEQUENCE_LENGTH, ENCODER_HIDDEN_STATES_DIM),
            None,
            None,
            None,
            None,
            {'text_embeds': torch.randn(BATCHSIZE,TEXT_EMBEDS_DIM), 'time_ids': torch.randn(BATCHSIZE,TIME_IDS_SIZE)},
            None
        ),
        output_path=unet_path,
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids", "dummy3", "dummy4", "added_cond_kwargs"],
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "timestep": {0:"steps"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "text_embeds":{0:"batch_size"},
            "time_ids":{0:"batch_size"},
            "out_sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}
        },
        opset=opset,
        use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
    )
    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = os.path.dirname(unet_model_path)
    unet = onnx.load(unet_model_path)
    # clean up existing tensor files
    shutil.rmtree(unet_dir)
    os.mkdir(unet_dir)
    # collate external tensor files into one
    onnx.save_model(
        unet,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    return True # success


def export_onnx_text_encoder(pipeline, output_path, opset):
    # TEXT ENCODER
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    hidden_state_list = [f"hidden_states.{i}" for i in range(13)]
    onnx_export(
        pipeline.text_encoder,
        # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
        model_args=(text_input.input_ids.to(torch.int32), None, None, None, True),
        output_path=output_path / "text_encoder" / "model.onnx",
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"] + hidden_state_list,
        dynamic_axes={
            key: {0: "batch_size"} if key == "pooler_output" else {0: "batch_size", 1: "sequence_length"}
            for key in ["input_ids","last_hidden_state", "pooler_output"] + hidden_state_list
        },
        opset=opset,
    )
    return True # success


def export_onnx_text_encoder_2(pipeline, output_path, opset):        
    # TEXT ENCODER 2
    text_enc_2_path = output_path / "text_encoder_2" / "model.onnx"
    text_input = pipeline.tokenizer_2(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    hidden_state_list = [f"hidden_states.{i}" for i in range(33)]
    onnx_export(
        pipeline.text_encoder_2,
        # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
        model_args=(text_input.input_ids.to(torch.int64), None, None, None, True),
        output_path=text_enc_2_path,
        ordered_input_names=["input_ids"],
        output_names=["text_embeds", "last_hidden_state"] + hidden_state_list,
        dynamic_axes={
            key: {0: "batch_size", 1: "sequence_length"}
            for key in ["input_ids", "text_embeds", "last_hidden_state"] + hidden_state_list
        },
        opset=opset,
    )
    text_enc_2_path = str(text_enc_2_path.absolute().as_posix())
    text_enc_2_dir = os.path.dirname(text_enc_2_path)
    text_enc_2 = onnx.load(text_enc_2_path)
    # clean up existing tensor files
    shutil.rmtree(text_enc_2_dir)
    os.mkdir(text_enc_2_dir)
    # collate external tensor files into one
    onnx.save_model(
        text_enc_2,
        text_enc_2_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    return True # success


@torch.no_grad()
def convert_models(output_path: str, 
                   model_path: str, 
                   opset: int = 16,
                   export_unet: bool = False,
                   export_text_encoder: bool = False,
                   export_text_encoder_2: bool = False,):
    pipeline = StableDiffusionXLPipeline.from_pretrained(model_path)
    output_path = Path(output_path)
    
    if export_text_encoder:
        start_time = time.perf_counter()
        export_onnx_text_encoder(pipeline, output_path, opset)
        print(f"Completed Text Encoder ONNX export in {time.perf_counter() - start_time:.2f} seconds")
    else:
        print(f"Not exporting Text Encoder ONNX file")
        
    if export_text_encoder_2:
        start_time = time.perf_counter()
        export_onnx_text_encoder_2(pipeline, output_path, opset)
        print(f"Completed Text Encoder 2 ONNX export in {time.perf_counter() - start_time:.2f} seconds")
    else:
        print(f"Not exporting Text Encoder 2 ONNX file")
    
    if export_unet:
        start_time = time.perf_counter()
        export_onnx_unet(pipeline, output_path, opset)
        print(f"Completed UNet ONNX export in {time.perf_counter() - start_time:.2f} seconds")
    else:
        print(f"Not exporting UNet ONNX file")

    
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate ONNX files for SDXL')
    parser.add_argument('output_path', type=str, help='ONNX output path')
    parser.add_argument('--model_path', type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help='Input model path')
    parser.add_argument('--opset', type=int, default=16, help='Opset version')
    parser.add_argument('--export_unet', action='store_true', help='Export UNet')
    parser.add_argument('--export_text_encoder', action='store_true', help='Export Text Encoder')
    parser.add_argument('--export_text_encoder_2', action='store_true', help='Export Text Encoder 2')

    return parser.parse_args()

if __name__ == "__main__":
    convert_models(**vars(parse_args()))
