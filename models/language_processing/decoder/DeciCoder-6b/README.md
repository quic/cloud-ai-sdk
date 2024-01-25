# Description

[DeciCoder-6b](https://huggingface.co/Deci/DeciCoder-6b) is a decoder-only large language model (LLM) developed by [Deci Ai](https://deci.ai) for code generation tasks. The architectures of the model was developed by AutoNAC which is Deci Ai's proprietary Neural Architecture Search technology. The model has a context length of 2048 tokens and is trained on the Python, Java, Javascript, C++, C#, Go, and Rust subsets of [The-Stack](https://huggingface.co/datasets/bigcode/the-stack) dataset. 

# Running on AIC100

## Available Compute Resources
The following cloud provider instances are equipped with AIC100 accelerators. 



|Provider | [AWS DL2q Instance](https://aws.amazon.com/ec2/instance-types/dl2q/) | [Cirrascale Instance](https://cirrascale.com/solutions-qualcomm-cloud-ai100.php) |
| --------------------- | --------------------- | -------------------------- |
|Cloud-AI Accelerators  |  8 Std (14 NSPs) SKUs | 1 to 8 Pro (16 NSPs) SKUs  |
|Supported Formats for [DeciCoder-6b](https://huggingface.co/Deci/DeciCoder-6b)| FP16 and [MX6](https://arxiv.org/abs/2302.08007)  | FP16 and [MX6](https://arxiv.org/abs/2302.08007) |

## Source of the Model

The model is downloaded from [HuggingFace](https://huggingface.co/Deci/DeciCoder-6b).
	
## Environment and Dependencies
Create Python virtual environment and activate.

        python3.8 -m venv llm_env
        source llm_env/bin/activate

Install the dependencies.

        python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
        python -m pip install onnx==1.15.0 onnxruntime==1.16.3 onnxsim==0.4.35 tiktoken==0.5.2 protobuf==3.20.2 urllib3==1.26.6 
        git clone --branch v4.35.2 --depth 1 https://github.com/huggingface/transformers
        cd transformers
        git apply ../Llama2_4.35.2.patch
        python -m pip install .
        cd ..

## Model and Hardware Parameters
Specify the model repo/name and the compilation parameters. Model will be compiled using MX6 compression. Let MX="" if you want to avoid MX6 compression. BS, PL and CL are Batchsize, Prompt Length and Context Length respectively. 

        MODEL_REPO="Deci"
        MODEL_NAME="DeciCoder-6b"
        BS=1
        PL=256
        CL=2048
        CORES=14
        MX="-mxfp6-matmul"

## Model Generation
Generate the model into onnx format.
		
        python generateModel.py --model-name ${MODEL_REPO}/${MODEL_NAME} --model-class LlamaForCausalLM

## Model Compilation for AIC100
Compile the onnx format into bin file. Modify BS, PL, CL, CORES, and MX if needed.

        bash compileModel.sh $MODEL_NAME $BS $PL $CL $CORES $MX

## Model Execution on AIC100
Run the bin file on AIC100. Modify DEVICE_ID if needed.

        PROMPT="insert your prompt here"
        DEVICE_ID=0
        python runModel.py --model-name ${MODEL_REPO}/${MODEL_NAME} --qpc ./qpc/${MODEL_NAME}-kv-${PL}pl-${CL}cl-${CORES}c${MX} --device_id $DEVICE_ID --prompt "${PROMPT}"

## References 
- [Shared Micro-exponents](https://arxiv.org/abs/2302.08007)

