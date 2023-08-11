# ResNext101 Model
---

## Source of the model
This model is adopted from [ResNext101](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).

## Description of the model
---

A residual neural network (ResNet) is an artificial neural network (ANN) of a kind that builds on constructs known from pyramidal cells in the cerebral cortex. Residual neural networks do this by utilizing skip connections, or shortcuts to jump over some layers.
ResNeXt is a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. While in ResNeXt, which increases cardinality, the different paths are merged by adding them together, and each path is similar.

Cardinality means the number of independent paths, to provide a new way of adjusting the model capacity.

## Framework and version
---

This model is in Pytorch and also exported to ONNX.
1. Pytorch CPU Version : 1.8.0
2. Python Version: 3.6.13

We use ModelPreparator tool to generate the ONNX model which is most optimized for AIC. ModelPreparator is part of QAIC-Pytools. So please make sure QAIC-Pytools is installed along with APPS SDK. This can be installed using command "sudo ./install.sh --enable-qaic-pytools" 
QAIC-Pytools folder can be found at: /opt/qti-aic/tools/qaic-pytools/

## Execution Steps
---

1. Execute resnet50_setup.sh
2. Execute the Script resnet50_run.sh


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.


