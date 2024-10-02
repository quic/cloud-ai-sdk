### Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
### SPDX-License-Identifier: BSD-3-Clause-Clear

# Instructions to run SD3 on Cloud AI 100

The instructions below are to run the [Stable Diffusion 3 model](stabilityai/stable-diffusion-3-medium) on Cloud AI 100. Compile time parameters may need to be adjusted for different cards and different SDKs.

## 1. Download model 

1. Setup environment varialble
```
mkdir cache
export HF_HOME=cache
```

2. Follow [instructions on HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)

## 2. Generate onnx files and compile for binaries

1.  Set up a virtual environment for ONNX generation and compilation
```
python3.8 -m venv env_onnx
source ./env_onnx/bin/activate
pip install -r requirements.txt
pip install wheel
```

2.  Create a folder for caching HuggingFace model downloads
```
mkdir compile_logs
mkdir qpc
touch run.sh
```

3. Install diffusers from source after patching for ONNX file generation
```
mkdir diffusers_onnx
cd diffusers_onnx
git clone --depth 1 --branch v0.30.0 https://github.com/huggingface/diffusers.git
cd diffusers
git apply --reject --whitespace=fix ../../patches/attention_patch.patch
pip install .
cd ../..
```

4. Install transformers from source (for T5 only)
```
git clone -b v4.41.2 https://github.com/huggingface/transformers.git
cd transformers
git apply --reject --whitespace=fix ../patches/transformer_patch.patch
pip install .
cd ..
```

5. Generate ONNX files and compile for binaries
```
bash run_config_gen.sh
```

## 3. Run the end-to-end SD3 inference

1. Set up a separate virtual environment 
```
python3.8 -m venv env_pipeline
source ./env_pipeline/bin/activate
pip install -r requirements.txt
pip install wheel
pip install --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
```

2.  Re-install diffusers from source after patching the SD3 pipeline for inference
```
mkdir diffusers_pipeline
cd diffusers_pipeline
git clone --depth 1 --branch v0.30.0 https://github.com/huggingface/diffusers.git
cd diffusers
git apply --reject --whitespace=fix ../../patches/pipeline_patch.patch
pip install .
cd ../..
```

3. Run the inference with 'sudo' flag if needed to access the AI 100 devices. 
```
sudo bash run_config_inference.sh
```

