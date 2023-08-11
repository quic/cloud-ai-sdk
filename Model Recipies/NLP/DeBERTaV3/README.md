# DebertaV3 Model
---

## Source of the model
This model is adopted from Hugging Face[DebertaV3](https://huggingface.co/microsoft/deberta-v3-xsmall).

## Description of the model
---

The DeBERTa V3 xsmall model comes with 12 layers and a hidden size of 384. It has only 22M backbone parameters with a vocabulary containing 128K tokens which introduces 48M parameters in the Embedding layer. This model was trained using the 160GB data as DeBERTa V2.

* deberta-v3-xsmall-classification.onnx
   1. Untrained Classification head. 

## Framework and version
---

*  This model is exported to ONNX
   1. ONNX-Runtime CPU Version : 1.8.1
   2. Python Version: 3.8.12
   3. Transoformers: 4.15.0


## Execution command
---

1. Open debertav3_run.sh and  Configure the following options as per your choice:

Select 'fp16' precision
precision='fp16'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

2. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.


