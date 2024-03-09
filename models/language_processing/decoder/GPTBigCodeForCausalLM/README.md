# GPTBigCodeForCausalLM 
Inference for models based on `GPTBigCodeForCausalLM` architecture can be executed on Cloud AI 100 platforms using the 3 steps - model generation, model compilation and model execution described below. Some examples of models supported by this recipe are 
    
	bigcode/tiny_starcoder_py
    bigcode/starcoderbase-1b
    bigcode/starcoderbase-3b
    bigcode/starcoderbase-7b
    bigcode/starcoder
    bigcode/starcoderplus
    bigcode/gpt_bigcode-santacoder


## Dependencies and versions

- Python 3.8.16
- Pytorch < 2
- Transformers 4.31.0
- ONNX 1.14.0
- ONNX Runtime 1.15.1
- ONNX Simplifier 0.4.31
- protobuf==3.20.2

## Model generation

1. Install requirements:

    python3.8 -m venv ./llm_env
    source ./llm_env/bin/activate
    pip install -r requirements.txt

2. Install patched tranformers library:

    git clone --branch v4.36.2 --depth 1 https://github.com/huggingface/transformers
    cd transformers
    git apply ../GPTBigCodeForCausalLM.patch
    pip install .
    cd ..
        
3. Specify the MODEL_NAME, MODEL_REPO, AUTH_TOKEN, BS (batch size), PL (prompt length), CL (context length), CORES (per SOC), DEVICE_ID, and SOCS in the init.sh. Let SOCS=4 or 1 if targetting Ultra or single devices. Let FORMAT="fp16" for fp16 format.

    MODEL_REPO=bigcode/starcoder
    MODEL_NAME=starcoder
    AUTH_TOKEN=your_unique_hf_auth_token
    BS=1
    PL=256
    CL=1024
    CORES=16
    DEVICE_ID=0
    SOCS=1
    FORMAT=mx6

5. Generate the model into onnx format. Then, skip to step 8 for benchmarking only. 
		
    python generateONNX.py --model-name $MODEL_REPO --model-class AutoModelForCausalLM --auth-token $AUTH_TOKEN

6. Model compilation.
	
    bash compileModel.sh $MODEL_NAME $BS $PL $CL $CORES $SOCS $FORMAT
        
7. Model execution.

    python runModel.py --model-name $MODEL_REPO --qpc ./qpc/${MODEL_NAME}-${BS}bs-${PL}pl-${CL}cl-$((CORES*SOCS))c-${FORMAT} --device_id $DEVICE_ID --token $AUTH_TOKEN --prompt "Hello, my name is Sarah and"


### Model support  
  
#### Tested Configurations (bigcode/starcoder) for AWS DL2q Instance 
|# Parameters | Ctx_len  | seq_len aka input-len | Batch-Size | MX6/FP16 | 
| ------ | ------------- | ------------- | ----------------- | -------- | 
|15B | 8192  | 1  | 14 | mx6 | 
|15B | 2048  | 1  | 56 | mx6 | 
|15B | 1024  | 1  | 84 | mx6 | 

## References 
- [LlamaForCausal execution on Cloud AI 100](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Model-Architecture-Support/Large-Language-Models/llm/)
    - [Precision - FP16 and MX6](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Model-Architecture-Support/Large-Language-Models/llm/#compile-the-model)
- [Shared Micro-exponents](https://arxiv.org/abs/2302.08007)
