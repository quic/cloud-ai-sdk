
### Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
### SPDX-License-Identifier: BSD-3-Clause-Clear


# Instructions to run SDXL on Cloud AI100 (PRO SKU)

The instructions below are to run [the Stable Diffusion XL model](stabilityai/stable-diffusion-xl-base-1.0) on Cloud AI100 PRO SKU. For other cards such as STD SKU, the block size in attention_processor.py will need to be tuned. Other compile time parameters may also need to be adjusted (e.g., -aic-num-cores)


## Folder Structure

```
├── README.md                               # This file
├── onnx_gen_utils.py                       # Utility module for ONNX generation
├── generate_onnx_files.py                  # Python script to generate ONNX files
├── fix_vae_decoder_onnx.py                 # Python script to fix VAE decoder precision issues
├── compile_models.sh                       # Script to compile all the model ONNX files
├── attention_patch.patch                   # Diffusers patch for ONNX generation
├── pipeline_patch.patch                    # Diffusers patch for SDXL inference
├── run_sdxl.py                             # End-to-end SDXL inference script running on AI100
```


## 1. Install Platform and Apps SDK

Install the latest Platform and Apps SDKs following the instructions in the [Cloud AI100 documentation](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/).


## 2. Generate ONNX files (skip this step if ONNX files are already available)

1.  Set up a virtual environment for ONNX generation
```
python3.8 -m venv env_sdxl_onnxgen
source ./env_sdxl_onnxgen/bin/activate
pip install networkx==3.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx==1.12.0 onnxruntime accelerate transformers
```

2.  Create a folder for caching HuggingFace model downloads, and export the environment variable HF_HOME
```
mkdir hf_home
export HF_HOME=$(pwd)/hf_home
```

3. Install diffusers from source after patching for ONNX file generation
```
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git
cd diffusers
git apply --reject --whitespace=fix ../attention_patch.patch
pip install .
cd ..
```

4. Generate ONNX files for text encoder and text encoder 2
```
python generate_onnx_files.py onnx_files/ --export_text_encoder --export_text_encoder_2
```

5. Generate ONNX files for UNet for batchsize = 2
```
python generate_onnx_files.py onnx_files/unet_bs2/ --export_unet
```

6. Modify attention_processor.py with blocksize = 256 directly in the installed diffusers library
```
sed -i 's/query_block_size = 128/query_block_size = 256/g' ./env_sdxl_onnxgen/lib/python3.8/site-packages/diffusers/models/attention_processor.py
```

7. Generate ONNX files for UNet for batchsize = 1
```
python generate_onnx_files.py onnx_files/unet_bs1/ --export_unet
```

8. Clone SDXL repository from HuggingFace to get the VAE Decoder ONNX file
```
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 HF_repo/stabilityai/stable-diffusion-xl-base-1.0
cd HF_repo/stabilityai/stable-diffusion-xl-base-1.0
git lfs pull -I vae_decoder/model.onnx
rm -rf .git/lfs # optional to save space
cd ../../../
```

9. Fix the FP16 overflow error in `vae_decoder` part of the model
```
python fix_vae_decoder_onnx.py --scaling-factor 128 --model-path ./HF_repo/stabilityai/stable-diffusion-xl-base-1.0/vae_decoder/model.onnx
mkdir onnx_files/vae_decoder/
mv ./HF_repo/stabilityai/stable-diffusion-xl-base-1.0/vae_decoder/model_fixed_128.onnx ./onnx_files/vae_decoder/
```


## 3. Compile the models (skip this step if QPCs are already available)
```
bash -x compile_models.sh
```


## 4. Run the end-to-end SDXL inference script

1. Set up a separate virtual environment for running SDXL (this would conflict with code changes for ONNX generation - hence separate environment)
```
python3.8 -m venv env_sdxl_infer
source ./env_sdxl_infer/bin/activate
pip install networkx==3.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx==1.12.0 onnxruntime accelerate transformers
pip install --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
```

2. Export HF_HOME environment variable again
```
export HF_HOME=$(pwd)/hf_home
```

3.  Re-install diffusers from source after patching the SDXL pipeline for inference
```
rm -rf diffusers
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git
cd diffusers
git apply --reject --whitespace=fix ../pipeline_patch.patch
pip install .
cd ..
```

4. Run the SDXL inference with 'sudo' flag if needed to access the AI100 devices. The `text_encoder`, `text_encoder_2`, and `vae_decoder` binaries are the same whether running on 1 device or 2 devices. You can pass in `--use_latents latents.pt` argument to use a pre-defined latents PyTorch file


For running on 1 device, pass in `device_id` argument, and `unet` argument point to the batchsize=2 QPC
```
sudo ./env_sdxl_infer/bin/python run_sdxl.py \
    --prompt <prompt_in_quotes> \
    --negative_prompt <negative_prompt_in_quotes> \
    --device_id 0 \
    --text_encoder ./qpc/text_encoder/ \
    --text_encoder_2 ./qpc/text_encoder_2/ \
    --unet ./qpc/unet-bs2/ \
    --vae_decoder ./qpc/vae_decoder/
```


For running on 2 devices, pass in `device_id` and `device_id2` arguments, and `unet` argument point to the batchsize=1 QPC
```
sudo ./env_sdxl_infer/bin/python run_sdxl.py \
    --prompt <prompt_in_quotes> \
    --negative_prompt <negative_prompt_in_quotes> \
    --device_id 0 \
    --device_id2 1 \
    --text_encoder ./qpc/text_encoder/ \
    --text_encoder_2 ./qpc/text_encoder_2/ \
    --unet ./qpc/unet-bs1/ \
    --vae_decoder ./qpc/vae_decoder/
```

