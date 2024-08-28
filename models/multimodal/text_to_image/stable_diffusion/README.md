## Description
---
Stable Diffusion is a latent text-to-image diffusion model that can generate photo-realistic images given any text input. 
## Source of the models
---
The models are downloaded from (https://huggingface.co)

## Environment and dependancies
---
```commandline
python3.8 -m venv stable_diffusion_env
source stable_diffusion_env/bin/activate
pip install torch==2.0.0 onnx==1.12.0 onnxruntime accelerate
```
Clone, patch and install transformerts

```commandline
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 65b5035a1d9b9b117ca5fd9ff33f5863408106a5 
git apply ../clip_text_encoder_changes.patch
pip install .
cd ..
```
Clone, patch and install diffusers
```commandline
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout d2a5247a1f919e1dfbf2a15f2fda21a1ea11a116
touch src/diffusers/aic_utils.py
git apply --reject --whitespace=fix ../sd_optimizations_v2.patch
pip install .
cd ..
```
## Generate the stable_diffusion_v1.5 onnx
```commandline
python generateModels.py --model_path runwayml/stable-diffusion-v1-5 --output_path stable_diffusion_v1.5_onnx
```
## AIC100 Compilation Commands
Create AIC Path

```commandline
mkdir stable_diffusion_AIC
```
Text Encoder
```commandline
rm -rf ./stable_diffusion_AIC/text_encoder/
/opt/qti-aic/exec/qaic-exec -m=stable_diffusion_v1.5_onnx/text_encoder/model.onnx -aic-hw -onnx-define-symbol=batch,1 -convert-to-fp16 -onnx-define-symbol=sequence,77 -aic-binary-dir=./stable_diffusion_AIC/text_encoder/ -aic-num-cores=1 -compile-only
```
Safety Checker (Optional)
```commandline
rm -rf ./stable_diffusion_AIC/safety_checker
/opt/qti-aic/exec/qaic-exec -m=./stable_diffusion_v1.5_onnx/safety_checker/model.onnx -aic-hw -onnx-define-symbol=batch,1 -convert-to-fp16 -onnx-define-symbol=channels,3 -onnx-define-symbol=height,512 -onnx-define-symbol=width,512 -onnx-define-symbol=clip_height,224 -onnx-define-symbol=clip_width,224 -aic-binary-dir=./stable_diffusion_AIC/safety_checker -compile-only
```
VAE Encoder (Required only for inpainting - skip this step for text to image)
```commandline
rm -rf ./stable_diffusion_AIC/vae_encoder/
/opt/qti-aic/exec/qaic-exec -m=stable_diffusion_v1.5_onnx/vae_encoder/model.onnx -aic-hw -onnx-define-symbol=batch,1 -convert-to-fp16 -onnx-define-symbol=sequence,77 -onnx-define-symbol=channels,3  -onnx-define-symbol=height,512  -onnx-define-symbol=width,512 -aic-binary-dir=./stable_diffusion_AIC/vae_encoder/ -aic-num-cores=1 -compile-only
```
VAE Decoder
```commandline
rm -rf ./stable_diffusion_AIC/vae_decoder/
/opt/qti-aic/exec/qaic-exec -m=stable_diffusion_v1.5_onnx/vae_decoder/model.onnx -aic-hw -aic-hw-version=2.0 -onnx-define-symbol=batch,1 -stats-batchsize=1  -convert-to-fp16 -onnx-define-symbol=sequence,77 -onnx-define-symbol=channels,4  -onnx-define-symbol=height,64  -onnx-define-symbol=width,64 -aic-binary-dir=./stable_diffusion_AIC/vae_decoder -aic-num-cores=14 -aic-num-of-instances=1 -multicast-weights -compile-only -aic-enable-depth-first -aic-depth-first-mem=32
```
Unet
```commandline
rm -rf ./stable_diffusion_AIC/unet/
/opt/qti-aic/exec/qaic-exec -m=stable_diffusion_v1.5_onnx/unet/model.onnx -aic-hw -aic-hw-version=2.0 -onnx-define-symbol=batch,2 -stats-batchsize=2 -convert-to-fp16 -onnx-define-symbol=sequence,77 -onnx-define-symbol=channels,4  -onnx-define-symbol=height,64  -onnx-define-symbol=width,64 -aic-binary-dir=./stable_diffusion_AIC/unet/ -aic-num-cores=12 -aic-num-of-instances=1 -mos=2 -ols=1    -compile-only
```
## Directory Preparation

Copy components from onnx path to AIC path
```commandline
cp -r stable_diffusion_v1.5_onnx/* stable_diffusion_AIC/
```
Replace json file in the AIC path
```commandline
cp model_index.json stable_diffusion_AIC/
```
## Run the pipeline
```commandline
sudo python aic_inference_dps.py --model_directory stable_diffusion_AIC/ --prompt "glass of wine by the beach"
```

