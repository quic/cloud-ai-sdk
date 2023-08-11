# EfficientNet B0 - B7
---

## Source of the model

This model is adopted from [efficientNet](https://github.com/lukemelas/EfficientNet-PyTorch).
It has all the 8 variants B0 - B7. 

EfficientNet PyTorch is a PyTorch re-implementation of EfficientNet. It is consistent with the original TensorFlow implementation, such that it is easy to load weights from a TensorFlow checkpoint. Further this pytorch model is exported to onnx.

---

## Description of the model

All the needed description of this model is available at [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch). This implementation is based on the paper [EfficientNet Paper](https://arxiv.org/abs/1905.11946).

The ONNX models (exported from pytorch) runs on AIC100 successfully.

## Execution Steps
---

1. Execute efficientnet_setup.sh

2. Open efficientnet_setup.sh and  Configure the following options as per your choice:

Select either 'int8' or 'fp16' precision
precision='int8'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

3. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.


