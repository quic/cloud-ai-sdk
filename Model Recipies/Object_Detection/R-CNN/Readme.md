# Mask RCNN
---

## Source of the model

<!-- https://jira-legal.qualcomm.com/jira/browse/OSRQCT-11028 -->
This model is adopted from [Detectron2](https://github.com/facebookresearch/detectron2). Instruction to setup Detectron2 in cpu.

This model follows the architecture described in this https://github.com/facebookresearch/detectron2 performs object-detection on 80 classes of COCO dataset. The related blog link for understating of model. Detectron2 is state-of-the-art model by FAIR, is faster than the previous-version i.e Detectron1. This supports pytorch framework includes more features such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, etc.

Detectron2 is implemented in pytorch framework, but it uses some operators from caffe2 as custom operators. These custom operators are exported in caffe and onnx format using custom exporter.

1) The exported onnx model from detectron2 contains caffe2 operators which are dynamic is nature. We removed the dynamicity of those operators and supported those operators. The following block diagram shows the flow and working of respective non-dynamic operators.

Callflow of caffe2 operators

2) The model outputs floating-point soft-masks of shape [num_classes x 28 x 28], this matches the description of model in original Mask-RCNN paper but not the pytorch detectron2 output. 


## Framework and version
---

This model is in Pytorch.
1. Pytorch Version : 1.6.0 [Link](https://pytorch.org/get-started/previous-versions/)
2. Python Version: 3.8.5

## Execution Steps
---

1. Run rcnn_setup.sh

2. Open rcnn_setup.sh and  Configure the following options as per your choice:

Select either 'mp' or 'fp16' precision
precision='mp'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

2. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.


