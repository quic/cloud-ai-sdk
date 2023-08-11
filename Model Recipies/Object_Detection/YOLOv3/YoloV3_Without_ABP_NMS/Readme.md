# YoloV3_Without_ABP_NMS
---

## Source of the model
This model is exported based on the repo [YoloV3](https://github.com/ultralytics/yolov3/tree/archive).

## Description of the model
---

 A new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

The part of the model follows the architecture described in this [YoloV3](https://arxiv.org/abs/1804.02767).
1. no Dynamic tensors
2. no control flow OPs

## Framework and version
---

This model is in Pytorch.
1. Pytorch Version : 1.7.0 [Link](https://pytorch.org/get-started/previous-versions/)
2. Python Version: 3.8.5

## Modifications done to the model (if any)
---

The original model repo have model in pytorch and the post processing part is seperated from Model Architecture before ABP NMS part.[here](https://github.com/ultralytics/yolov3/blob/1be31704c9c690929e4f6e6d950f40755ef2dcdc/models/yolo.py#L54).

Below are the steps taken to generate the Models:
1. We modified the code from [here](https://github.com/ultralytics/yolov3/) to remove the ABP post process part and NMS layers and generate the ONNX and Torchscript model for Input shapes [1, 3, 320, 416], [1, 3, 416, 416], [1, 3, 608, 608], [1, 3, 608, 800], [1, 3, 640, 640] and [1, 3, 640, 1152].

## Execution Steps
---

1.Open yolov3_run.sh and  Configure the following options as per your choice:

Select either 'int8' or 'fp16' precision
precision='int8'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

2. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.

