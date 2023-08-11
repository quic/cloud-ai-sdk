# DistilRoBERTa Model
---

## Source of the model
This model is adopted from Hugging Face [DistilRoberta-trained QnA head](https://huggingface.co/twmkn9/distilroberta-base-squad2) and [DistilRoberta-untrained QnA head](https://huggingface.co/distilroberta-base/blob/main/config.json) with Question Answering Head.

## Description of the model
---

This model is a distilled version of the RoBERTa-base model. It follows the same training procedure as DistilBERT. The code for the distillation process can be found here. This model is case-sensitive: it makes a difference between english and English.

## Frameworks required
---

This model is exported to ONNX
1. ONNX 
2. ONNX Runtime 
3. onnx-simplifier 
4. Python Version: 3.8.10
5. Pytorch 
6. Optimum

We use ModelPreparator tool to generate the ONNX model which is most optimized for AIC. ModelPreparator is part of QAIC-Pytools. So please make sure QAIC-Pytools is installed along with APPS SDK. This can be installed using command "sudo ./install.sh --enable-qaic-pytools" 
QAIC-Pytools folder can be found at: /opt/qti-aic/tools/qaic-pytools/

## Execution command
---

1. Execute distillroberta_setup.sh
 
2. Open distillroberta_run.sh and  Configure the following options as per your choice:

Select 'fp16' precision
precision='fp16'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

2. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.



