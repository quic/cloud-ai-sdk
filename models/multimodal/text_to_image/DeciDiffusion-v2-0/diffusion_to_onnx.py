# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from diffusers.models.attention_processor import AttnProcessor
from onnx import numpy_helper
from torch.onnx import export

import onnx
from diffusers import OnnxStableDiffusionPipeline, StableDiffusionPipeline
from diffusers.pipelines.onnx_utils import OnnxRuntimeModel
from packaging import version

is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")


def modify_config_to_support_aic(config_path: str):
    aic_config_path = config_path[:-len('.json')] + '_aic' + '.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    for model in ['unet', 'text_encoder', 'vae_encoder', 'vae_decoder']:
        assert config[model][0] == 'diffusers'
        assert config[model][1] == 'OnnxRuntimeModel'
        config[model][1] = 'AICRuntimeModel'  # override

    if config['requires_safety_checker']:
        assert config['safety_checker'][0] == 'diffusers'
        assert config['safety_checker'][1] == 'OnnxRuntimeModel'
        config['safety_checker'][1] = 'AICRuntimeModel'  # override

    if config['scheduler'][1] == 'SqueezedDPMSolverMultistepScheduler':
        config['scheduler'][0] = 'diffusers'

    with open(aic_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def fix_onnx_fp16(gen_models_path, model_base_name):
    finfo = np.finfo(np.float16)
    fp16_max = finfo.max
    fp16_min = finfo.min
    model = onnx.load(f"{gen_models_path}/{model_base_name}.onnx")
    fp16_fix = False
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        nptensor = numpy_helper.to_array(tensor, gen_models_path)
        if nptensor.dtype == np.float32 and (np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)):
            # print(f'tensor value : {nptensor} above {fp16_max} or below {fp16_min}')
            nptensor = np.clip(nptensor, fp16_min, fp16_max)
            new_tensor = numpy_helper.from_array(nptensor, tensor.name)
            tensor.CopyFrom(new_tensor)
            fp16_fix = True

    onnx.load_external_data_for_model(model, gen_models_path)
    size_gb = model.ByteSize() / 1073741824
    if fp16_fix:
        # Save FP16 model
        print("Found constants out of FP16 range, clipped to FP16 range")
        # model_base_name += "_fixed_for_fp16"
        if size_gb <= 2:
            onnx.save(model, f=f"{gen_models_path}/{model_base_name}.onnx")
        else:
            onnx.save(model, f=f"{gen_models_path}/{model_base_name}.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
                      location=f"{model_base_name}.onnx.data", convert_attribute=True)

        print(f"Saving modified onnx file at {gen_models_path}/{model_base_name}.onnx")

    return model_base_name


def onnx_export(
        model,
        model_args: tuple,
        output_path: Path,
        ordered_input_names,
        output_names,
        dynamic_axes,
        opset,
        use_external_data_format=False,
):
    if hasattr(model, 'set_attn_processor'):
        print(f"Executing {model.__class__.__name__}.set_attn_processor(AttnProcessor())")
        model.set_attn_processor(AttnProcessor())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
    # so we check the torch version for backwards compatibility
    if is_torch_less_than_1_11:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            enable_onnx_checker=True,
            opset_version=opset,
        )
    else:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset,
        )


@torch.no_grad()
def convert_models(model_path: str, output_path: str, opset: int):
    print(model_path)
    latent_dim = 64
    pixel_dim = 8 * latent_dim
    text_encoder_dim = 768
    use_safety_checker = True
    if 'DeciDiffusion' not in model_path:
        if 'stable-diffusion-v1-5' in model_path:
            assert 'runwayml/' not in model_path, "Will mess up future hierarchy"
            model_path = 'runwayml/' + 'stable-diffusion-v1-5'

        if 'stable-diffusion-2-1' in model_path:
            latent_dim = 96
            text_encoder_dim = 1024
            use_safety_checker = False  # not supported
            assert 'stabilityai/' not in model_path, "Will mess up future hierarchy"
            model_path = 'stabilityai/' + 'stable-diffusion-2-1'
        pipeline = StableDiffusionPipeline.from_pretrained(model_path)
    else:
        if not model_path.startswith("Deci/"):  # Minor UX
            model_path = "Deci/" + model_path
        pipeline = StableDiffusionPipeline.from_pretrained(model_path, custom_pipeline=model_path)
        pipeline.unet = pipeline.unet.from_pretrained(model_path, subfolder="flexible_unet")

    num_params = round(sum(parameter.numel() for parameter in pipeline.unet.parameters()) / 1e6, 3)
    print(f"UNet #params = {num_params} M")
    output_path = Path(output_path)
    # TEXT ENCODER
    print("Exporting text_encoder...")
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    onnx_export(
        pipeline.text_encoder,
        # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
        model_args=text_input.input_ids.to(torch.int32),
        output_path=output_path / "text_encoder" / "model.onnx",
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        },
        opset=opset,
    )
    del pipeline.text_encoder

    # UNET
    print("Exporting unet...")
    unet_path = output_path / "unet" / "model.onnx"
    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(2, pipeline.unet.in_channels, latent_dim, latent_dim),
            torch.LongTensor([0, 1]),
            torch.randn(2, 77, text_encoder_dim),
            False,
        ),
        output_path=unet_path,
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
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
    del pipeline.unet

    # VAE ENCODER
    print("Exporting vae_encoder...")
    vae_encoder = pipeline.vae
    # need to get the raw tensor output (sample) from the encoder
    vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(sample, return_dict)  # [0].sample()
    onnx_export(
        vae_encoder,
        model_args=(torch.randn(1, 3, pixel_dim, pixel_dim), False),
        output_path=output_path / "vae_encoder" / "model.onnx",
        ordered_input_names=["sample", "return_dict"],
        output_names=["latent_sample"],
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )

    # VAE DECODER
    print("Exporting vae_decoder...")
    vae_decoder = pipeline.vae
    # forward only through the decoder part
    vae_decoder.forward = vae_encoder.decode
    onnx_export(
        vae_decoder,
        model_args=(torch.randn(1, 4, latent_dim, latent_dim), False),
        output_path=output_path / "vae_decoder" / "model.onnx",
        ordered_input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )
    del pipeline.vae

    # SAFETY CHECKER
    if use_safety_checker:
        print("Exporting safety_checker...")
        safety_checker = pipeline.safety_checker
        safety_checker.forward = safety_checker.forward_onnx
        onnx_export(
            pipeline.safety_checker,
            model_args=(torch.randn(1, 3, 224, 224), torch.randn(1, pixel_dim, pixel_dim, 3)),
            output_path=output_path / "safety_checker" / "model.onnx",
            ordered_input_names=["clip_input", "images"],
            output_names=["out_images", "has_nsfw_concepts"],
            dynamic_axes={
                "clip_input": {0: "batch", 1: "channels", 2: "clip_height", 3: "clip_width"},
                "images": {0: "batch", 1: "height", 2: "width", 3: "channels"},
            },
            opset=opset,
        )
    del pipeline.safety_checker

    onnx_pipeline = OnnxStableDiffusionPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_encoder"),
        vae_decoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_decoder"),
        text_encoder=OnnxRuntimeModel.from_pretrained(output_path / "text_encoder"),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
        scheduler=pipeline.scheduler,
        safety_checker=OnnxRuntimeModel.from_pretrained(output_path / "safety_checker") if use_safety_checker else None,
        feature_extractor=pipeline.feature_extractor,
    )

    onnx_pipeline.save_pretrained(output_path)
    print("ONNX pipeline saved to", output_path)

    del pipeline
    del onnx_pipeline
    _ = OnnxStableDiffusionPipeline.from_pretrained(output_path, provider="CPUExecutionProvider")
    print("ONNX pipeline is loadable")

    # FIX FP16
    for model in ["text_encoder", "unet", "vae_encoder", "vae_decoder", "safety_checker"]:
        print(f"Fixing FP16: {model}")
        fix_onnx_fp16(output_path, model + '/model')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="DeciDiffusion-v2-0",
        choices=['DeciDiffusion-v1-0', 'DeciDiffusion-v2-0', 'stable-diffusion-v1-5', 'stable-diffusion-2-1'],
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--opset",
        default=14,
        type=int,
        help="The version of the ONNX operator set to use.",
    )

    args = parser.parse_args()

    convert_models(args.model_path, args.output_path, args.opset)
    modify_config_to_support_aic(os.path.join(args.output_path, 'model_index.json'))
