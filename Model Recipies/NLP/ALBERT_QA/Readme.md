# ALBERT Model
---

## Source of the model
- This model is adopted from Hugging Face [albert-base-v2](https://huggingface.co/albert-base-v2) with Question Answering Head.
- Added another albert model from Hugging face, finetuned on SquadV2. This model is adopted from [twmkn9/albert-base-v2-squad2](https://huggingface.co/twmkn9/albert-base-v2-squad2)

## Description of the model
---

ALBERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. The model onboarded here uses albert-base-v2 with Question Answering Head attached to it.

## Framework and version
---

````
This model is exported to ONNX
1. ONNX : 1.10.1
2. ONNX Runtime : 1.8.1
3. onnx-simplifier : 0.3.6
4. Python Version: 3.8.11
5. Pytorch Version: 1.9.1
6. Transoformers: (https://github.com/huggingface/transformers.git)
````




## Execution command
---

1) run albert_setup.sh to download the pretrained ALBERT model.
2) open the albert_run.sh  and configure the config and precision that you need.
	available options for config : "throughput" or "latency"
	available options for precision : "fp16"  or "mixed"
3) Now run albert_run.sh using the below command.
	"source albert_run.sh"


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.

