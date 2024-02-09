# MptForCausalLM 
Inference for models (eg. MPT-7B) based on `MptForCausalLM` architecture can be executed on Cloud AI 100 platforms using the 3 steps - model generation, model compilation and model execution described below. 

## Dependencies and versions

- Python 3.8.16
- Pytorch 2.0.1
- Transformers 4.32.0
- ONNX 1.14.0
- ONNX Runtime 1.15.1
- ONNX Simplifier 0.4.31
- protobuf==3.20.2
- urllib3==1.26.6

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
	git apply --reject --whitespace=fix ../MptForCausalLM.patch
	pip install -e .
    cd ..
    ```

3. Please refer to LlamaForCausalLM recipe for instructions along with relevant scripts for model generation, compilation, and execution for `MptForCausalLM` class of models. Use `MptForCausalLM` instead of `LlamaForCausalLM` in model generation and model execution stages. 

