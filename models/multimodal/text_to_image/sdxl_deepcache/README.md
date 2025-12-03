### Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
### SPDX-License-Identifier: BSD-3-Clause-Clear

# Instructions to run SDXL on Cloud AI 100 with DeepCache

The instructions below are to run the [Stable Diffusion XL model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) with [DeepCache](https://github.com/horseee/DeepCache) on Cloud AI 100.


## Pre-requisites

Install the moreutils package for the `ts` timestamp tool:
```
sudo apt update
sudo apt-get install moreutils
```

Install Git Large File System (LFS) support

```
sudo apt update
sudo apt-get install git-lfs
```

## 1. Generate onnx files and compile for binaries

1.  Set up a virtual environment for ONNX generation and compilation
```
python3.10 -m venv env_onnx
source ./env_onnx/bin/activate
pip install -r requirements.txt
```

2.  Setup environments
```
mkdir cache
mkdir qpc
mkdir compile_logs
```

3. Install diffusers from source after patching for ONNX file generation
```
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers_onnx
cd diffusers_onnx
git apply --reject --whitespace=fix ../patches/attention_patch.patch
pip install .
cd ..
```

4. Install DeepCache for ONNX file generation (deep UNet) 
```
git clone https://github.com/horseee/DeepCache.git
cd DeepCache
git apply --reject --whitespace=fix ../patches/deepcache_unet.patch
pip install .
cd ..
```

5. Prepare VAE Decoder 
```
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 cache/stabilityai/stable-diffusion-xl-base-1.0
cd cache/stabilityai/stable-diffusion-xl-base-1.0
git lfs pull -I vae_decoder/model.onnx
rm -rf .git/lfs # optional to save space
cd ../../../
```

6. Generate ONNX files and compile for binaries
```
touch run.sh
bash run_config_deep.sh
```

7. Modify the UNet to be the shallow version
```
sed -i '963s/False/True/' env_onnx/lib/python3.10/site-packages/DeepCache/sdxl/unet_2d_condition.py
```

8. Generate ONNX file and compile shallow UNet for DeepCache
```
bash run_config_shallow.sh
```

## 2. Run the end-to-end SDXL inference

1. Set up a separate virtual environment for running SDXL 
```
python3.10 -m venv env_pipeline
source ./env_pipeline/bin/activate
pip install -r requirements.txt
pip install --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
```

2.  Re-install diffusers and DeepCache from source after patching the SDXL pipeline for inference
```
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers_pipeline
cd diffusers_pipeline
pip install .
cd ..
```

3. Install DeepCache and prepare the pipeline for inference
```
git clone https://github.com/horseee/DeepCache.git deepcache_pipeline
cd deepcache_pipeline
git apply --reject --whitespace=fix ../patches/deepcache_pipeline.patch
pip install .
cd ..
```

4. Run the SDXL inference with 'sudo' flag if needed to access the AI100 devices. 
```
sudo bash run_config_inference.sh $(which python3)
```
Note: ```CACHE_INTERVAL``` variable in ```run_config_inference.sh``` refers to the period of caching

