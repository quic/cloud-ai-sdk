# BertLarge Model
---

## Source of the model
This model is adopted from [BertLarge](https://github.com/mlcommons/inference/tree/master/language/bert).

## Description of the model
---

Bidirectional Encoder Representations from Transformers is a Transformer-based machine learning technique for natural language processing pre-training developed by Google.

## Framework and version
---

This model is in Pytorch and also exported to both TorchScript and ONNX
1. Pytorch CPU Version : 1.4.0
2. TensorFlow CPU Version : 2.3.0
3. ONNX CPU Version : 1.8.0
4. Python Version: 3.3

## Modifications done to the model (if any)
--

1. We have created script [Model.py] to save the Model in ONNX and torchScript format and inference the standalone Pytorch and ONNX models using there native framworks.1

## Execution command
---

1. Execute bert_setup.sh
 
2. Open bert_run.sh and  Configure the following options as per your choice:

Select 'fp16' or 'mp' precision
precision='fp16'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

2. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.



