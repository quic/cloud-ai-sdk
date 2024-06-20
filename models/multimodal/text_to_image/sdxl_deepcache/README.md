# Instructions to run SDXL on Cloud AI100 with DeepCache

The instructions below are to run the [Stable Diffusion XL model](stabilityai/stable-diffusion-xl-base-1.0) on Cloud AI100. 


## 1. Generate onnx files and compile for binaries

1.  Set up a virtual environment for ONNX generation and compilation
```
python3.8 -m venv env_onnx
source ./env_onnx/bin/activate
pip install networkx==3.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx==1.12.0 onnxruntime accelerate transformers
```

2.  Setup environments
```
mkdir cache
mkdir qpc
mkdir compile_logs
```

3. Install diffusers from source after patching for ONNX file generation
```
mkdir diffusers_onnx
cd diffusers_onnx
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git
cd diffusers
git apply --reject --whitespace=fix ../../patches/attention_patch.patch
pip install .
cd ../..
```

4. Prepare VAE Decoder 
```
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 cache/stabilityai/stable-diffusion-xl-base-1.0
cd cache/stabilityai/stable-diffusion-xl-base-1.0
git lfs pull -I vae_decoder/model.onnx
rm -rf .git/lfs # optional to save space
cd ../../../
```

5. Install DeepCache for ONNX file generation (deep UNet) 
```
pip install DeepCache==0.1.1
cp patches/unet_2d_condition_deep.py env_onnx/lib/python3.8/site-packages/DeepCache/sdxl/unet_2d_condition.py
```

6. Generate ONNX files and compile for binaries
```
touch run.sh
bash run_config_deep.sh
```

7. Modify the UNet to be the shallow version
```
cp patches/unet_2d_condition_shallow.py env_onnx/lib/python3.8/site-packages/DeepCache/sdxl/unet_2d_condition.py
```

8. Generate ONNX file and compile shallow UNet for DeepCache
```
bash run_config_shallow.sh
```

## 2. Run the end-to-end SDXL inference

1. Set up a separate virtual environment for running SDXL 
```
python3.8 -m venv env_pipeline
source ./env_pipeline/bin/activate
pip install networkx==3.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx==1.12.0 onnxruntime accelerate transformers
pip install --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
```

2.  Re-install diffusers and DeepCache from source after patching the SDXL pipeline for inference
```
mkdir diffusers_pipeline
cd diffusers_pipeline
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git
cd diffusers
pip install .
cd ../..
```

3. Install DeepCache and prepare the pipeline for inference
```
pip install DeepCache==0.1.1
cp patches/pipeline_stable_diffusion_xl.py env_pipeline/lib/python3.8/site-packages/DeepCache/sdxl/pipeline_stable_diffusion_xl.py
```

4. Run the SDXL inference with 'sudo' flag if needed to access the AI100 devices. 
```
sudo bash run_config_inference.sh
```
Note: ```CACHE_INTERVAL``` variable in ```run_config_inference.sh``` refers to the period of caching

