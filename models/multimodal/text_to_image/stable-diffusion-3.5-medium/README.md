### Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
### SPDX-License-Identifier: BSD-3-Clause-Clear

# Instructions to run SD3.5 on Cloud AI 100

The instructions below are to run the [Stable Diffusion 3.5 model](stabilityai/stable-diffusion-3.5-medium) on Cloud AI 100. Compile time parameters may need to be adjusted for different cards and different SDKs.

## 1. Download model

1. Setup environment varialble
```
mkdir cache
export HF_HOME=cache
export HF_TOKEN=<your-huggingface-auth-token>
```

2. Follow [instructions on HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) to gain access to model.

## 2. Generate onnx files and compile for binaries

1.  Set up a virtual environment for ONNX generation and compilation
```
python3.10 -m venv env_onnx
source ./env_onnx/bin/activate
pip install -r requirements.txt
```

2.  Create a folder for caching HuggingFace model downloads
```
mkdir compile_logs
mkdir qpc
touch run.sh
```

3. Install diffusers from source after patching for ONNX file generation
```
git clone --depth 1 --branch v0.31.0 https://github.com/huggingface/diffusers.git diffusers-onnx
cd diffusers-onnx
git apply --reject --whitespace=fix ../patches/attention_patch.patch
pip install .
cd ..
```

4. Install transformers from source (for T5 text_encoder_3 only)
```
git clone -b v4.41.2 https://github.com/huggingface/transformers.git transformers-dev
cd transformers-dev
git apply --reject --whitespace=fix ../patches/transformer_patch.patch
pip install .
cd ..
```

5. Generate ONNX files and model binaries
```
bash run_config_gen.sh
```

## 3. Run the end-to-end SD3 inference

1. Set up a separate virtual environment
```
python3.10 -m venv env_pipeline
source ./env_pipeline/bin/activate
pip install -r requirements.txt
pip install --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
```

2.  Re-install diffusers from source after patching the SD3 pipeline for inference
```
git clone --depth 1 --branch v0.31.0 https://github.com/huggingface/diffusers.git diffusers-pipeline
cd diffusers-pipeline
git apply --reject --whitespace=fix ../patches/pipeline_patch.patch
pip install .
cd ..
```

3. Run the inference with 'sudo' flag if needed to access the AI 100 devices.
```
sudo bash run_config_inference.sh "<positive prompt>" "<negative prompt>"
```

## 4. Python interface

```
source ./env_pipeline/bin/activate
```

```python
from model import QAICStableDiffusion3

model = QAICStableDiffusion3()
prompt = 'photo of 8k ultra realistic harbour, port, boats, sunset, beautiful light, golden hour, full of colour, cinematic lighting, battered, trending on artstation, 4k, hyperrealistic, focused, extreme details, cinematic, masterpiece'
image = model.generate(prompt, guidance=7.0)[0]
image.save('harbor.png')
```

