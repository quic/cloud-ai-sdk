# Instructions to run SDXL-Turbo on Cloud AI 100

The instructions below are to run the [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) model on Cloud AI 100. Compile time parameters may need to be adjusted for different cards and different SDKs.

## Pre-requisites

Install the moreutils package for the `ts` timestamp tool:
```
sudo apt-get install moreutils
```

## 1. Generate onnx files and compile for binaries

1.  Set up a virtual environment for ONNX generation and compilation
```
python3.10 -m venv env_onnx
source ./env_onnx/bin/activate
pip install -r requirements.txt
pip install wheel
```

2.  Create a folder for caching Hugging Face model downloads, and export the environment variable HF_HOME
```
mkdir cache
mkdir compile_logs
mkdir qpc
touch run.sh
export HF_HOME=${PWD}/cache
```

3. Install diffusers from source after patching for ONNX file generation
```
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers-onnx
cd diffusers-onnx
git apply --reject --whitespace=fix ../patches/attention_patch.patch
pip install .
cd ..
```

4. Prepare VAE Decoder
```
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/stabilityai/sdxl-turbo cache/stabilityai/sdxl_turbo
cd cache/stabilityai/sdxl_turbo
git lfs pull -I vae_decoder/model.onnx
rm -rf .git/lfs # optional to save space
cd ../../../
```

5. Generate ONNX files and compile for binaries
```
bash run_config_gen.sh
```

## 2. Run the end-to-end SDXL-Turbo inference

1. Set up a separate virtual environment for running SDXL Turbo
```
python3.10 -m venv env_pipeline
source ./env_pipeline/bin/activate
pip install -r requirements.txt
pip install wheel
pip install --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
```

2.  Re-install diffusers from source after patching the SDXL Turbo pipeline for inference
```
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers-pipeline
cd diffusers-pipeline
git apply --reject --whitespace=fix ../patches/pipeline_patch_separate.patch
pip install .
cd ..
```

4. Run the SDXL-Turbo inference with 'sudo' flag if needed to access the AI 100 devices.
```
sudo bash run_config_inference.sh
```

## 3. Run an OpenAI-compatible REST endpoint

```
source ./env_pipeline/bin/activate
python3 server.py
```

Test the endpoint:

```
curl http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test-key' \
  -d '{
    "model": "sdxl-turbo",
    "prompt": "photo of 8k ultra realistic harbour, port, boats, sunset, beautiful light, golden hour, full of colour, cinematic lighting, battered, trending on artstation, 4k, hyperrealistic, focused, extreme details, cinematic, masterpiece",
    "n": 1,
    "size": "512x512",
    "response_format": "b64_json"
  }'
```
