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
Clone, patch and install transformer

```commandline
git clone https://github.com/huggingface/transformers.git transformers-dev
cd transformers-dev
git checkout 65b5035a1d9b9b117ca5fd9ff33f5863408106a5
git apply ../clip_text_encoder_changes.patch
pip install .
cd ..
```
Clone, patch and install diffusers
```commandline
git clone https://github.com/huggingface/diffusers.git diffusers-dev
cd diffusers-dev
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

## Compile the models
```
bash -x compile_models.sh
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

The script creates Prompt1_aic.png with the generated image.

