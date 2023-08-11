# GPT2
---

## Source of the model
* [GPT2-Small](https://huggingface.co/gpt2).

## Description of the model
---

GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was trained to guess the next word in sentences. 
* GPT2-Small model has 12 layers, 768 dimension and 12 heads, totalizing 124M parameters and a vocabulary size of 50,257.

## Framework and version
---

* This model is exported to ONNX
    1. Python Version: 3.8
    2. Pytorch Version: 1.9.1
    3. Transoformers: 4.10.2
    4. ONNX-Runtime CPU Version : 1.8.1


## Execution command
---

1. Open gpt2-run.sh and  Configure the following options as per your choice:

Select 'fp16' precision
precision='fp16'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

2. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.


