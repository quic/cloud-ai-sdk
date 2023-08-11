# YOLOv5 Models
---

## Source of the model
These models are adopted from [YOLOv5](https://github.com/ultralytics/yolov5).

## Description of the model
---

* YOLO model is a fast compact object detection model that is very performant relative to its size and it has been steadily improving.
* YOLOv5 is the first of the YOLO models to be written in the PyTorch framework and it is much more lightweight and easy to use.
* That said, YOLOv5 did not make major architectural changes to the network in YOLOv3. Here, we specifically used c5ba2abb4afb9fe8c671f14eb5200647893efe30 commit id from Yolov5 Ultralytics repo. Also, we specifically use [release 5.0](https://github.com/ultralytics/yolov5/releases/tag/v5.0) checkpoints for exporting the final model.

## Framework and version
---

This model is in Pytorch and also exported to ONNX.
* Pytorch CPU Version : 1.8.0
* Python Version: 3.6.13


## Limitations with the model generation scripts:
---
Please note that these scripts would work with v5.0 release of Ultralytics source repo. Exporting using any other release would require changes in the scripts. 


## Execution Steps
---

1.Open yolov5_run.sh and  Configure the following options as per your requirements:

Select either 'int8' or 'fp16' precision
precision='int8'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

2. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.

