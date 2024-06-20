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
from diffusers import AutoPipelineForText2Image
from DeepCache.sdxl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline as DeepCacheStableDiffusionXLPipeline
from onnx import external_data_helper, numpy_helper
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from onnx_generation.onnx_gen_utils import onnx_export
from utils import scale_conv


def export_onnx_vae(
        pipeline, 
        output_path, 
        opset,
        block_size = None,
        image_size = None,
        latent_size = None,
        vae_type = 'vae',
    ):
    if vae_type == 'tiny':
        print("Generating tiny vae...")
        from diffusers import AutoencoderTiny
        pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl")

        vae_path = output_path / f"vae_decoder_{block_size}b_{image_size}i_tiny" / "model.onnx"
        vae_decoder = pipeline.vae
        vae_decoder.forward = pipeline.vae.decode
        onnx_export(
            vae_decoder,
            model_args=(torch.randn(1, 4, latent_size, latent_size)),
            output_path=vae_path,
            ordered_input_names=["latent_sample"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch_size", 1: "num_channels_latent", 2: "height_latent", 3: "width_latent"},
            },
            opset=opset,
        )
    else:
        print("Generating sdxl vae...")
        # fix fp16 overflow issue
        scaling_factor = 128
        model_path = "./cache/stabilityai/stable-diffusion-xl-base-1.0/vae_decoder/model.onnx"
        model = onnx.load(model_path)
        scale_conv(model, "/decoder/up_blocks.2/upsamplers.0/conv/Conv", scaling_factor)
        scale_conv(model, "/decoder/up_blocks.3/resnets.0/conv2/Conv", scaling_factor)
        # scale_conv(model, "/decoder/up_blocks.3/resnets.0/conv_shortcut/Conv", scaling_factor)
        scale_conv(model, "/decoder/up_blocks.3/resnets.1/conv2/Conv", scaling_factor)
        scale_conv(model, "/decoder/up_blocks.3/resnets.2/conv2/Conv", scaling_factor)
        output_path_fix = model_path[:-5] + f"_fixed_{scaling_factor}.onnx"
        onnx.save(model, output_path_fix)

        vae_path = os.path.join(output_path, f"vae_decoder_{block_size}b_{image_size}i_vae", "model.onnx")
        os.makedirs(vae_path[:-10], exist_ok=True)
        shutil.copyfile(output_path_fix, vae_path)

    return True


def export_onnx_unet(
        pipeline, 
        output_path, 
        opset,
        image_size = None,
        block_size = None,
        latent_channels = 4,
        latent_size = None,
        sequence_length = 77,
        encoder_hidden_states_dim = None,
        text_embeds_dim = None,
        time_ids_size = None,
        unet_type = 'deep',
    ):
    # UNET
    unet_path = output_path / f"unet_{block_size}b_{image_size}i_dc_{unet_type}" / "model.onnx"
    # batch_size=2 to avoid collapse of batch dimension in ONNX UNet tracing
    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(2, latent_channels, latent_size, latent_size),
            torch.randint(0, 20, (2,), dtype=torch.int64), # random int64
            torch.randn(2, sequence_length, encoder_hidden_states_dim),
            torch.randn(2, 320, 128, 128),
            None,
            None,
            None,
            None,
            {'text_embeds': torch.randn(2, text_embeds_dim), 'time_ids': torch.randn(2, time_ids_size)},
            None
        ),
        output_path=unet_path,
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "replicate_prv_feature", "text_embeds", "time_ids", "dummy3", "dummy4", "added_cond_kwargs"],
        output_names=["out_sample", "replicate_prv_feature_RetainedState"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {0: "batch_size", 1: "latent_channels", 2: "height", 3: "width"},
            "timestep": {0:"steps"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "replicate_prv_feature": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "text_embeds":{0:"batch_size"},
            "time_ids":{0:"batch_size"},
            "out_sample": {0: "batch_size", 1: "latent_channels", 2: "height", 3: "width"},
            "replicate_prv_feature_RetainedState": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        },
        opset=opset,
        use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
    )
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(f"onnx_files/unet_{block_size}b_{image_size}i_dc_{unet_type}/model.onnx")
    print([ort_session.get_outputs()[i].name for i in range(len(ort_session.get_outputs()))])

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

    # Generate custom-IO files
    with open(f"onnx_files/custom_io_{unet_type}.yaml", "w") as fp:
        if unet_type != "deep":
            fp.write("# Model Inputs\n\n")
            fp.write(f" - IOName: replicate_prv_feature\n   Precision: float16\n\n")
        fp.write("# Model Outputs\n\n")
        fp.write(f" - IOName: replicate_prv_feature_RetainedState\n   Precision: float16\n\n")
    return True # success


def export_onnx_text_encoder(
        pipeline, 
        output_path, 
        opset,
        block_size = None,
        image_size = None,
    ):
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
        output_path=output_path / f"text_encoder_{block_size}b_{image_size}i" / "model.onnx",
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"] + hidden_state_list,
        dynamic_axes={
            key: {0: "batch_size"} if key == "pooler_output" else {0: "batch_size", 1: "sequence_length"}
            for key in ["input_ids","last_hidden_state", "pooler_output"] + hidden_state_list
        },
        opset=opset,
    )
    return True # success


def export_onnx_text_encoder_2(
        pipeline, 
        output_path, 
        opset,
        block_size = None,
        image_size = None,
    ):        
    # TEXT ENCODER 2
    text_enc_2_path = output_path / f"text_encoder_2_{block_size}b_{image_size}i" / "model.onnx"
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
def convert_models(
        output_path: str, 
        model_path: str, 
        opset: int = 16,
        image_size = None,
        block_size_deep = 128,
        block_size_shallow = 128,
        latent_size = None,
        seq_len = None,
        unet_type = 'deep',
        vae_type = 'vae',
        latent_channels = None,
        encoder_hidden_states_dim = None,
        text_embeds_dim = None,
        time_ids_size = None,
        export_unet: bool = False,
        export_text_encoder: bool = False,
        export_text_encoder_2: bool = False,
        export_vae: bool = False,
        together: bool = False,
    ):
    print(f"Model: {model_path}")
    pipeline = AutoPipelineForText2Image.from_pretrained(model_path)
    output_path = Path(output_path)
    
    if not together:
        if export_text_encoder:
            start_time = time.perf_counter()
            export_onnx_text_encoder(pipeline, output_path, opset, image_size=image_size, block_size=block_size_deep)
            print(f"Completed Text Encoder ONNX export in {time.perf_counter() - start_time:.2f} seconds")
        else:
            print(f"Not exporting Text Encoder ONNX file")
            
        if export_text_encoder_2:
            start_time = time.perf_counter()
            export_onnx_text_encoder_2(pipeline, output_path, opset, image_size=image_size, block_size=block_size_deep)
            print(f"Completed Text Encoder 2 ONNX export in {time.perf_counter() - start_time:.2f} seconds")
        else:
            print(f"Not exporting Text Encoder 2 ONNX file")
        
        if export_unet:
            start_time = time.perf_counter()
            block_size = block_size_deep if unet_type == "deep" else block_size_shallow
            pipeline_dc = DeepCacheStableDiffusionXLPipeline.from_pretrained(model_path)
            export_onnx_unet(
                    pipeline_dc, 
                    output_path, 
                    opset,
                    image_size = image_size,
                    block_size = block_size,
                    latent_channels = latent_channels,
                    latent_size = latent_size,
                    sequence_length = seq_len,
                    encoder_hidden_states_dim = encoder_hidden_states_dim,
                    text_embeds_dim = text_embeds_dim,
                    time_ids_size = time_ids_size,
                    unet_type = unet_type,
            )
            print(f"Completed UNet ONNX export in {time.perf_counter() - start_time:.2f} seconds")
        else:
            print(f"Not exporting UNet ONNX file")

        if export_vae:
            start_time = time.perf_counter()
            export_onnx_vae(
                    pipeline, 
                    output_path, 
                    opset,
                    block_size = block_size_deep,
                    image_size = image_size,
                    latent_size = latent_size,
                    vae_type = vae_type,
            )
            print(f"Completed VAE ONNX export in {time.perf_counter() - start_time:.2f} seconds")
        else:
            print(f"Not exporting VAE ONNX file")
    else:
        raise NotImplementedError

