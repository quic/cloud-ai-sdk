# YOLOv5 Models
---

## **Source of the model**
These models are adopted from [YOLOv7](https://github.com/WongKinYiu/yolov7).

## **Description of the model**
---

* YOLOV7 model by AlexeyAB and WongKinYiu is one of the latest addition to yolo object detector model's family with best accuracy and performance. YOLOV8 is also released by Ultralytics few days after the release of this model.
* YOLOv7 have different variants with improved performance and accuracy for edge devices(tiny yolov7), gpus(yolov7) and cloud devices(yolov7-e6e) .
* The major architectural change compared to previous yolo models is that it uses model re-parameterization, E-ELAN(Extended efficient layer aggregation networks) which reduces the number of parameters and activations for better accuracy and  performance.


## **Framework and version**
---

This model is in Pytorch and also exported to ONNX.

* Pytorch CPU Version : 1.9.1
* Python Version: 3.8.0



## Execution Steps
---

1.Run yolov7_setup.sh to download and convert the yolov7 model and convert to ONNX
2.Open yolov5_run.sh and  Configure the following options as per your requirements:

Select either 'int8' or 'fp16' precision
precision='int8'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

3. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.


