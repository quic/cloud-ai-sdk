# SSD MobileNet V1
---


## Source of the model
---

This model (commitID: 1eb0c5c506a3f685a74abd0a52d31678b812f4fd) is taken from [SSD](https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py) class of [MLCommons](https://github.com/mlcommons/inference) github repo.
The datasets used are [COCO] (https://cocodataset.org/#download).


## Description of the model
---

SSDMobileNetV1: A Deep Neural Network for Object Detection.
This model follows the architecture described in this [MobileNetV1 Paper](https://arxiv.org/abs/1704.04861).
>
1.  No Dynamic tensors
2.  1 Dynamic OP: Non-Zero.
3.  No control flow OPs


## Framework and version
---

This model is in Pytorch (Official github repo). Exported it to ONNX and TorchScript.
>
1.  Pytorch Version : 1.8.0
2.  ONNX Version: 1.8.0
3.  ONNX-RUNTIME Version : 1.7.0
4.  Python Version: 3.6.12

We use ModelPreparator tool to generate the ONNX model which is most optimized for AIC. ModelPreparator is part of QAIC-Pytools. So please make sure QAIC-Pytools is installed along with APPS SDK. This can be installed using command "sudo ./install.sh --enable-qaic-pytools" 
QAIC-Pytools folder can be found at: /opt/qti-aic/tools/qaic-pytools/


## Execution command
---

1. Execute setup_ssdmobilenet.sh
 
2. Execute run_ssdmobilenet.sh 



## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.


