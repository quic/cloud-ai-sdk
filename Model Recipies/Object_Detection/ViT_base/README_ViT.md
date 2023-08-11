# Vision Transformer(ViT) Model
---

## Source of the model
This model is adopted from [Vision Transformer](https://huggingface.co/google/vit-base-patch16-224).

## Description of the model
---


The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. Next, the model was fine-tuned on ImageNet (also referred to as ILSVRC2012), a dataset comprising 1 million images and 1,000 classes, also at resolution 224x224. In this, the images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.
![Architecture](./INTERNAL_USE/Architecture.PNG)



## Framework and version
---

This model is in Pytorch .pth format.
1. Python Version: 3.6.13
2. Pytorch CPU Version : 1.8.0

## Modifications done to the model (if any)
---

The script will download the Vision Transformer(ViT) model weights from HuggingFace repository, construct the model from definition and convert the same into ONNX model and TorchScript model.

## Execution Steps
---

1.Open vitbase_run.sh and  Configure the following options as per your choice:

Select 'fp16' precision
precision='fp16'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

2. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.


