# Speculative decoding - CodeGen

CodeGen is an autoregressive language model for program synthesis trained sequentially on The Pile, BigQuery, and BigPython.

Cloud AI 100 supports speculative decoding. Speculative decoding has two parts to it, Target language model (TLM) and Draft language model (DLM) that belong to the model family. Inferencing can be executed on Cloud AI 100 platforms using the 3 steps - model generation, model compilation and model execution as described below.

## Dependencies and versions

- Python 3.8.16
- Pytorch 2.0.1
- Transformers 4.33.2
- ONNX 1.14.0
- ONNX Runtime 1.15.1
- ONNX Simplifier 0.4.31
- protobuf==3.20.2
- tiktoken

## Setup and install requirements

1. **Setup temporary and cache directories:**
	```
	mkdir <workspace-path>/tmp
	export TMPDIR=<workspace-path>/tmp
	export PIP_CACHE_DIR=<workspace-path>/tmp
	export TRANSFORMERS_CACHE=<workspace-path>/tmp/hub
	export HF_HOME=<workspace-path>/tmp
	```

2. **Install requirements:**
	```
	python3.8 -m venv ./llm_env
	source ./llm_env/bin/activate
	pip install -r requirements.txt
	```

## Model generation 

1. **Install patched tranformers library:**
	```
	git clone --branch v4.33.2 --depth 1 https://github.com/huggingface/transformers
	cd transformers
	git apply ../CodeGen.patch
	pip install .
	cd ..
	```

2. **Run the generation script:** <br>
 - **Generating Draft language model (DLM) onnx:**
	
	```
	MODEL_REPO="Salesforce"
	MODEL_NAME="codegen-350M-mono"
	
	sudo python generateONNX.py --model-name ${MODEL_REPO}/${MODEL_NAME} --model-class AutoModelForCausalLM
	```

	Example:

	```
	sudo python generateONNX.py --model-name Salesforce/codegen-350M-mono --model-class AutoModelForCausalLM
	```

	Model will be generated in the `codegen-350M-mono-kv` subdirectory.

- **Generating Target language model (TLM) onnx:** <br>
	**a) Copy over the tlm_modeling_codegen.py file provided to the transformers package folder:** <br>
		
	```
	cd transformers
	cp ./src/transformers/models/codegen/modeling_codegen.py ./backup_orig_modeling_codegen.py
	cp ../tlm_modeling_codegen.py ./src/transformers/models/codegen/modeling_codegen.py
	pip install .
	cd ..
	```

	**b) Run the Target language model generation script:**
	
	```
	MODEL_REPO="Salesforce"
	MODEL_NAME="codegen-6B-mono"

	sudo python generateONNX.py --model-name ${MODEL_REPO}/${MODEL_NAME} --model-class AutoModelForCausalLM
	```

	Example:

	```
	sudo python generateONNX.py --model-name Salesforce/codegen-6B-mono --model-class AutoModelForCausalLM
	```

	Model will be generated in the `codegen-6B-mono-kv` subdirectory.

## Model compilation

1. **Modify the custom_io.yaml files:**
- In both the subdirectories generated above `codegen-350M-mono-kv` and `codegen-6B-mono-kv`, modify the custom_io.yaml files by adding below lines, where IOName "logits" can be found from the respective onnx models.
- Make sure the indentation is correct when modifing the custom_io.yaml files.
		
	```
	- IOName: logits
	  Precision: float16
	```

2. **Run the compilation script: <br>
	a) Draft language model (DLM):** <br>

	```
	bash compileModel.sh --model-name codegen-350M-mono --dlm --num-cores <1-16 nsp> --<mx6 or fp16> --num-devices <number-of-devices>
	```

	Example: <br>

	```
	bash compileModel.sh --model-name codegen-350M-mono --dlm --num-cores 4 --fp16 --num-devices 1
	```

	**b) Target language model (TLM):** <br>

	```
	bash compileModel.sh --model-name codegen-6B-mono --spec-length <spec_len> --tlm --num-cores <1-16 nsp> --<mx6 or fp16> --num-devices <number-of-devices>
	```

	Example: <br>

	```
	bash compileModel.sh --model-name codegen-6B-mono --spec-length 7 --tlm --num-cores 10 --mx6 --num-devices 1
	```

	This will compile the models and place the generated model binaries in the `qpc` subdirectory.

	**Note:**
	- Use '14' (#nsp) for DL2q instance at AWS. '16' (#nsp) can be used at Cirrascale instances.
	- --num-devices: if the number of devices are more than one (Example: --num-devices 4) then multi-device configuration is used and during run-time one need to pass `--mq` argument and device-ids, example "--device-ids 0,1,2,3" for 4 devices.
	- Creating or modifying network specialization JSON files:
		- Currently, only batch_size = 1 is supported with Speculative Decoding.
		- To change the prompt length(seq_len), context length(ctx_len), speculation length(k), etc. edit the DLM and TLM specialization files:
			1. **dlm_specializations.json:** <br>
				- The specialization with seq_len=prompt length is for prefill,
				- Then next with seq_len=2 (aways set to 2 regarless of value of 'k') for parallel KV cache update & decode.
				- Last one with seq_len=1 (aways set to 1 regarless of value of 'k') for decode.
				- Modify the context length(ctx_len) as required, with the same value for all three specialization.
			2. **tlm_specializations_k.json:** <br>
				- Ensure that the name of the file has a "_k" where k should be speculation length (i.e. tlm_specializations_7.json for speculation length = 7).
				- The first specialization is for prefill, the second and the third, for parallel evaluation of Draft tokens.
				- Modify the context length(ctx_len) as required, with the same value for all three.
				- Set the seq_len to prompt length for the first specialization.
				- Set the seq_len to speculation length (i.e., k) for the second and
				- Speculation length + 1 (i.e., k+1) for the third specialization.

		**Note:**
		batch_size and ctx_len should be identical for a model across all its specializations for both DLM and TLM.
		

		Examples of JSON files for DLM and TLM network specialization can be found in the `network-specializations` sub-directory.

## Model execution
**Run the runSpeculativeDecoding.sh script:**
		
```
bash runSpeculativeDecoding.sh --model-repo <model-repo-name> --tlm-model-name <tlm-model-name> --dlm-model-name <dlm-model-name> --tlm-precision <mx6 or fp16> --dlm-precision <mx6 or fp16> --num-cores-tlm <1-16> --num-cores-dlm <1-16> --pl <prompt-length> --cl <context-length> --spec-length <spec-length> --mq --tlm-device-ids <device-id(s)> --dlm-device-ids <device-id(s)> --model-family <model-family-name> --prompt-file <path-to-text-prompt-file>
```

Example: <br>

```
sudo bash runSpeculativeDecoding.sh --model-repo Salesforce --tlm-model-name codegen-6B-mono --dlm-model-name codegen-350M-mono --tlm-precision mx6 --dlm-precision fp16 --num-cores-dlm 4 --num-cores-tlm 10 --pl 32 --cl 256 --spec-length 7 --tlm-device-ids 0 --dlm-device-ids 0 --model-family codegen --prompt-file ./codegen_prompts.txt
```

**Note:** <br>
- For multi-device configuration, pass `--mq` argumnet as well as number of device ids based on compile configuration, example "--tlm-device-ids 0,1,2,3 and --dlm-device-ids 0,1,2,3" for 4 devices in the above run command.
- For any specific prompt, create a new input prompt text file or enter it in the example (codegen_prompts.txt) file. Make sure prompt ends with `:<endofprompt>` as shown in the example file.
