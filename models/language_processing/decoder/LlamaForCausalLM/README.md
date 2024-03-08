# LlamaForCausalLM 
Inference for models based on `LlamaForCausalLM` architecture can be executed on Cloud AI 100 platforms using the 3 steps - model generation, model compilation and model execution described below. 

## Dependencies and versions

- Python 3.8.16
- Pytorch 2.0.1
- Transformers 4.31.0
- ONNX 1.14.0
- ONNX Runtime 1.15.1
- ONNX Simplifier 0.4.31
- protobuf==3.20.2

## Model generation

1. Install requirements:

    ```
    python3.8 -m venv ./llm_env
    source ./llm_env/bin/activate
    pip install -r requirements.txt
    ```

2. Install patched tranformers library:

    ```
    git clone --branch v4.32.0 --depth 1 https://github.com/huggingface/transformers
    cd transformers
    git apply ../LlamaForCausalLM.patch
    pip install .
    cd ..
    ```

3. Run the generation script:
    - `python generateONNX.py --model-name <Model_Path> --model-class LlamaForCausalLM`
    - Example: `python generateONNX.py --model-name LLM360/Amber --model-class LlamaForCausalLM`

Models will be generated in the `Amber-kv` subdirectory.

## Model compilation

### Single SoC 
This applies to the Cloud AI 100 [Standard, Pro and the Ultra SKUs](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Architecture/#cloud-ai-platforms). 

**Note: To change the seq_len, context_len, batch_size, etc. edit the specializations.json file <br> The first section is for prefill and the second is for decode. <br>`batch_size` and `ctx_len` should be identical for a model. <br>`seq_len` aka `prompt_length` may be a power of 2 for the prefill stage and 1 for the decode stage.**

- Run the compilation script: 
    
    `bash compileModel_single_soc.sh <Path to model generated with generateONNX.py> [mx6 or fp16] [1-16 nsp]`<br>
    Example: `bash compileModel_single_soc.sh Amber-kv mx6 14`<br>
    Example: `bash compileModel_single_soc.sh Amber-kv fp16 14`<br>

    **Note: Use '14' (#nsp) for DL2q instance at AWS. '16' (#nsp) can be used at Cirrascale instances (Pro or Ultra)**

    This will compile the model and place the generated QPC(s) in `qpc` subdirectory.

    For higher batch size support, compile the model with compileModel_pl1.sh. Set the desired batch_size and ctx_len in the specialization_pl1.json file. Set seq_len = 1.  
    `bash compileModel_pl1.sh <Path to model generated with generateONNX.py> [mx6 or fp16] [1-16 nsp]`<br>
    Example: `bash compileModel_pl1.sh Amber-kv mx6 14`<br>
    Example: `bash compileModel_pl1.sh Amber-kv fp16 14`<br>

#### Model support  
  
| # Parameters  | AWS DL2q Instance - 8 Std (14 NSPs) SKU Accelerators| Cirrascale Instance - 1 to 8 Pro (16 NSPs) SKU Accelerators|
| ------------- | ------------- | ----------------- |
| 7B  | FP16 and MX6  | FP16 and MX6 |
| 13B  | MX6  | FP16 and MX6 |

#### Tested Configurations for AWS DL2q Instance 
|# Parameters | Ctx_len  | seq_len aka input-len | Batch-Size | MX6/FP16 | Comments |
| ------ | ------------- | ------------- | ----------------- | -------- | ------ |
|7B | 2048  | 256  | 1 | FP16 | use compileModel.sh |
|7B | 2048 | 256 | 1 | MX6 | use compileModel.sh | 
|7B | 2048 | 1 to 1536 | 1/2/4/8 | MX6 | use compileModel_pl1.sh |
|7B | 256 | 1 to 256 | 1/2/4/8/16/32/64 | MX6 | use compileModel_pl1.sh |
|7B | 256 | 1 to 256 | 1/2/4/8 | FP16| use compileModel_pl1.sh |
|7B | 1536 | 1 to 1536 | 1/2 | FP16| use compileModel_pl1.sh |
| 13B | 2048 | 256 | 1 | MX6 | use compileModel.sh |

### Multi-SoC
This applies to the Cloud AI 100 [Ultra SKU](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Architecture/#cloud-ai-platforms). 

- Run the compilation script 
    `bash compileModel_multi_soc.sh <Path to model generated with generateONNX.py> <BS> <PL> <CL> <CORES> <MOS> <OLS> <SOCS> <MX | default FP16>`
    Example: `sudo bash compileModel_multi_soc.sh Amber-kv 1 256 1024 16 16 1 4 -mxfp6-matmul`

    This will compile the models and place the generated model binaries in the `qpc` subdirectory.

	- `<model-name>` : Model name (Example: Amber-kv)
	- `<BS>` : Batch-Size
	- `<PL>` : Prompt length
	- `<CL>`	: Context length
	- `<CORES>` : Number of AI cores (For Ultra card - 16)
	- `<MOS>` : Effort level to reduce on-chip memory usage. Set MOS = CORES
	- `<OLS>`	: Factor to increase splitting of network operations for more fine-grained parallelism.
	- `<SOCS>` : Number of SoC's been used, Ultra Card has 4 SoC's.
	- `<MX>` : MX6 compression format use flag `-mxfp6-matmul`, keep it blank for FP16, which is default setting.

## Model execution

- Run the runModel.py script:

    ## Single SoC  
    `python runModel.py --model-name <Model_Path> --qpc <qpc/model-name-kv-compile_params> --device_id <Single QID#> --prompt <Enter prompt>` <br>
    Example: `python runModel.py --model-name LLM360/Amber --qpc qpc/Amber-kv-256pl-2048cl-14c --device_id 2 --prompt <Enter text within double-quotes seperated by | for BS greater than 1>` <br>

    ## Multi SoC 
    `python runModel.py --model-name <Model_Path> --qpc <qpc/model-name-kv-compile_params> --device_id <DEVICES> --prompt <Enter prompt>` <br>
    Example: `python runModel.py --model-name LLM360/Amber --qpc qpc/Amber-kv-256pl-2048cl-14c --device_id 0,1,2,3 --prompt <Enter text within double-quotes seperated by | for BS greater than 1>` <br>


    This will generate text with the compiled QPC's and finally print the output tokens/s generated by the compiled models. <br>
    runModel.py supports user prompt lenghts greater than the seq_len provided during model compilation. This feature is called input prompt chunking. 

## References 
- [LlamaForCausal execution on Cloud AI 100](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Model-Architecture-Support/Large-Language-Models/llm/)
    - [Precision - FP16 and MX6](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Model-Architecture-Support/Large-Language-Models/llm/#compile-the-model)
- [Shared Micro-exponents](https://arxiv.org/abs/2302.08007)
