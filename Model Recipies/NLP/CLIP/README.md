# CLIP (Contrastive Language–Image Pre-training) Model
---

## Source of the model
This model is adopted from Hugging Face [CLIP](https://huggingface.co/openai/clip-vit-base-patch16).

## Description of the model
---

CLIP is a zero-shot image classifier. CLIP architecture includes ViT (Vision transformer) and Text transformer.

## Framework and version
---

This model is exported to ONNX

1. ONNX-Runtime CPU Version : 1.9.0
2. Python Version: 3.8.11
3. Transoformers: 4.12.0

## Model Generation Steps (if any)
---

     To generate model with two outputs['logits per image','logits per text'] (This will take more than an hour to generate split file)


## Execution command
---

1. Execute clip_setup.sh
 
2. Open clip_run.sh and  Configure the following options as per your choice:

Select 'fp16' precision
precision='fp16'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

2. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.


