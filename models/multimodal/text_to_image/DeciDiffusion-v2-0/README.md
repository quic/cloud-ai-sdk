## Description
[DeciDiffusion v2.0](https://huggingface.co/Deci/DeciDiffusion-v2-0) is a 732 million parameter text-to-image diffusion model developed by [Deci Ai](https://deci.ai) that can generate photo-realistic images given any text input. The UNET architectures of the model was developed by AutoNAC which is Deci Ai's proprietary Neural Architecture Search technology. The model was trained on a curated subset of the LAION-Aesthetics dataset. 

## Available Compute Resources
The following cloud provider instances are equipped with AIC100 accelerators. 

|Provider | [AWS DL2q Instance](https://aws.amazon.com/ec2/instance-types/dl2q/) | [Cirrascale Instance](https://cirrascale.com/solutions-qualcomm-cloud-ai100.php) |
| --------------------- | --------------------- | -------------------------- |
|Cloud-AI Accelerators  |  8 Std (14 NSPs) SKUs | 1 to 8 Pro (16 NSPs) SKUs  |
|Supported Formats for [DeciDiffusion v2.0](https://huggingface.co/Deci/DeciDiffusion-v2-0)| FP16 and [MX6](https://arxiv.org/abs/2302.08007)  | FP16 and [MX6](https://arxiv.org/abs/2302.08007) |


## Source of the models
The models are downloaded from [HuggingFace](https://huggingface.co/Deci/DeciDiffusion-v2-0).

## Environment and dependencies
Create Python virtual environment and activate.
```commandline
python3.8 -m venv deci_diffusion_env
source deci_diffusion_env/bin/activate
```
Install the dependencies. Clone, patch and install transformers and diffusers. This is needed to support operators which are specific to AIC100.
```commandline
python -m pip install torch==2.0.0 onnx==1.12.0 onnxruntime accelerate
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 65b5035a1d9b9b117ca5fd9ff33f5863408106a5 
git apply ../clip_text_encoder_changes.patch
python -m pip install .
cd ..
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout v0.21.4
git apply --reject --whitespace=fix ../ai100_diffusers_v0.21.4.patch
python -m pip install .
cd ..
```
## Model Generation
Generate the deci_diffusion_v2.0 compnents into onnx formats.
```commandline
python diffusion_to_onnx.py --model_path DeciDiffusion-v2-0  --output_path ./onnx_files_DeciDiffusion-v2-0
```
## Model Compilation for AIC100
Create the AIC Path.
```commandline
rm -rf ./deci_diffusion_2.0_AIC
mkdir ./deci_diffusion_2.0_AIC
```
Compile the Text Encoder.
```commandline
rm -rf ./deci_diffusion_2.0_AIC/text_encoder/
/opt/qti-aic/exec/qaic-exec -m=./onnx_files_DeciDiffusion-v2-0/text_encoder/model.onnx -aic-hw -onnx-define-symbol=batch,1 -convert-to-fp16 -onnx-define-symbol=sequence,77 -aic-binary-dir=./deci_diffusion_2.0_AIC/text_encoder/ -aic-num-cores=1 -compile-only
```
Compile the VAE Encoder.
```commandline
rm -rf ./deci_diffusion_2.0_AIC/vae_encoder/
/opt/qti-aic/exec/qaic-exec -m=./onnx_files_DeciDiffusion-v2-0/vae_encoder/model.onnx -aic-hw -onnx-define-symbol=batch,1 -convert-to-fp16 -onnx-define-symbol=sequence,77 -onnx-define-symbol=channels,3  -onnx-define-symbol=height,512  -onnx-define-symbol=width,512 -aic-binary-dir=./deci_diffusion_2.0_AIC/vae_encoder/ -aic-num-cores=1 -compile-only
```
Compile the VAE Decoder.
```commandline
rm -rf ./deci_diffusion_2.0_AIC/vae_decoder/
/opt/qti-aic/exec/qaic-exec -m=./onnx_files_DeciDiffusion-v2-0/vae_decoder/model.onnx -aic-hw -aic-hw-version=2.0 -onnx-define-symbol=batch,1 -stats-batchsize=1  -convert-to-fp16 -onnx-define-symbol=sequence,77 -onnx-define-symbol=channels,4  -onnx-define-symbol=height,64  -onnx-define-symbol=width,64 -aic-binary-dir=./deci_diffusion_2.0_AIC/vae_decoder/ -aic-num-cores=12 -aic-num-of-instances=1 -multicast-weights -compile-only -allocator-dealloc-delay=4 -aic-enable-depth-first -aic-depth-first-mem=18
```
Compile the UNET.
```commandline
rm -rf ./deci_diffusion_2.0_AIC/unet/
/opt/qti-aic/exec/qaic-exec -m=./onnx_files_DeciDiffusion-v2-0/unet/model.onnx -aic-hw -aic-hw-version=2.0 -onnx-define-symbol=batch,2 -stats-batchsize=2 -convert-to-fp16 -onnx-define-symbol=sequence,77 -onnx-define-symbol=channels,4  -onnx-define-symbol=height,64  -onnx-define-symbol=width,64 -aic-binary-dir=./deci_diffusion_2.0_AIC/unet/ -aic-num-cores=12 -aic-num-of-instances=1 -mos=2 -ols=1    -compile-only
```
Compile the Safety Checker.
```commandline
rm -rf ./deci_diffusion_2.0_AIC/safety_checker/
/opt/qti-aic/exec/qaic-exec -m=./onnx_files_DeciDiffusion-v2-0/safety_checker/model.onnx -aic-hw -onnx-define-symbol=batch,1 -convert-to-fp16 -onnx-define-symbol=channels,3 -onnx-define-symbol=height,512 -onnx-define-symbol=width,512 -onnx-define-symbol=clip_height,224 -onnx-define-symbol=clip_width,224 -aic-binary-dir=./deci_diffusion_2.0_AIC/safety_checker -compile-only
```
## Directory Preparation
Copy components from onnx path to AIC path.
```commandline
cp -r ./onnx_files_DeciDiffusion-v2-0/feature_extractor ./deci_diffusion_2.0_AIC
cp -r ./onnx_files_DeciDiffusion-v2-0/scheduler ./deci_diffusion_2.0_AIC
cp -r ./onnx_files_DeciDiffusion-v2-0/tokenizer ./deci_diffusion_2.0_AIC
cp -r ./onnx_files_DeciDiffusion-v2-0/model_index_aic.json ./deci_diffusion_2.0_AIC/model_index.json
```
## Model Execution on AIC100
Run the pipeline.
```commandline
PROMPT="insert your prompt here"
DEVICE_ID=0
python aic_inference_example.py --aic_dir ./deci_diffusion_2.0_AIC --output_path ./ --repetitions 5 --prompt "${PROMPT}" --device_id $DEVICE_ID
```

## References 
- [Shared Micro-exponents](https://arxiv.org/abs/2302.08007)
