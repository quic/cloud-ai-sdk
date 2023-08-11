# YoloV4_Without_ABP_NMS
---

## Source of the model
This model is exported based on the repo [YoloV4](https://github.com/ultralytics/yolov3/tree/archive).

## Description of the model
---

 A new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

## Framework and version
---

This model is in Pytorch.
1. Pytorch Version : 1.7.0 [Link](https://pytorch.org/get-started/previous-versions/)
2. Python Version: 3.8.5

## Modifications done to the model (if any)
---

The original model repo have model in pytorch and the post processing part is seperated from Model Architecture as mentioned [here](https://github.com/ultralytics/yolov3/blob/archive/detect.py#L97).

Below are the steps taken to generate the Models:
1. We modified the code from [here](https://github.com/ultralytics/yolov3/tree/archive) to add the Detection and NMS layers and generate the Complete ONNX graph with NMS module for Input shapes [1, 3, 320, 416], [1, 3, 416, 416] , [1, 3, 640, 640] and [1, 3, 640, 800].
 
## Execution command
---

1. The Steps for generating the yoloV3_ABP_NMS models are below:
2. Install all the required packages in a fresh env with python3:
>
        conda create --name yolov4_env
        conda activate yolov4_env
        conda install python=3.6 pip numpy -y
        pip install -r requirements.txt
3. We need to clone the repo, download weights and follow the steps below to generate complete graph.
>
        git clone -b archive https://github.com/ultralytics/yolov3
        cd yolov3
        git apply --reject --whitespace=fix ./yoloV4WithoutModels.patch
        python3 detect.py --cfg cfg/yolov4.cfg --weights ./weightFiles/yolov4.weights

        # For Leaky Relu version
        python3 detect.py --cfg cfg/yolov4-relu.cfg --weights ./weightFiles/yolov4.weights

This will generate yolov4_416_416_without_ABP_NMS.onnx and yolov4_416_416_without_ABP_NMS.pt

4. This will generate the ONNX and Pytorch Trace script format models without NMS layers in it for given input resolution in the file, We can generate for different resolution by changes the input dimensions, Change the input dimention accordingly in the above script.

5. We can pass this generated models to preparator to validate and generate optimized graph out of it. See steps below:
>
        source /opt/qti-aic/dev/python/qaic-env/bin/activate
        python /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config yolov4_ultralytics_model_info_smart_nms.yaml

5.  Commands to run Yolov4 ONNX and pt models using qaic-exec on AIC H/W:

        Command for ONNX FP32 Version:/opt/qti-aic/exec/qaic-exec -m=./yolov4_416_416_without_ABP_NMS.onnx -input-list-file=./list.txt -aic-hw -aic-hw-version-2.0 -write-output-start-iter=1 -write-output-num-samples=1 -write-output-dir=./AICOutputs/ONNX

        Command for ONNX INT8 Version: /opt/qti-aic/exec/qaic-exec -m=./yolov4_416_416_without_ABP_NMS.onnx -input-list-file=./list.txt -aic-hw -aic-hw-version-2.0 -convert-to-quantize -quantization-schema=symmetric_with_uint8 -quantization-precision=Int8 -write-output-start-iter=1 -write-output-num-samples=1 -write-output-dir=./AICOutputs/ONNX

6.   Commands to run YoloV4 Pytorch Trace Script models using qaic-exec on AIC H/W:

        Command for Pytorch FP32 Version:/opt/qti-aic/exec/qaic-exec -m=./yolov4_416_416_without_ABP_NMS.pt -input-list-file=./list.txt -model-input=1,float,[1,3,416,416] -aic-hw -aic-hw-version=2.0 -write-output-start-iter=1 -write-output-num-samples=1 -write-output-dir=./AICOutputs/Pytorch

7.    We have also, provided the scripts to generate the raw file, infer and plot the bounding boxes after loading the AIC Outputs, Also there also already pre generated raw files for all three input resolution for sample image in inputFiles folder.
> 
        Sample Usage of Scripts: 
        1. Command to generate raw file for input [1, 3, 416, 416]
        python generateRawInput.py --image_path ./inputFiles/zidane.jpg --image_height_width 416 416

Note: Inference script will be soon uploaded here.
        
## Functional Support status
---

| Platform | Status |
| ----------- | ----------- |
| AIC100 | ONNX -- Yes
           Pytorch -- https://qti-jira.qualcomm.com/jira/browse/QRANIUMSW-5316
| Qognition | N/A |
| Makena | N/A |

## list of ONNX operators.
---

* Tanh
* SoftPlus
* Slice
* LeakyRelu
* MaxPool
* Transpose
* Shape
* Mul
* Concat
* Reshape
* Constant
* Resize
* Conv
* Add

## list of Pytorch Trace Script operators.
---

* aten::view
* aten::max_pool2d
* aten::contiguous
* aten::leaky_relu
* aten::softplus
* aten::upsample_nearest2d
* aten::_convolution
* aten::mul
* prim::ListConstruct
* aten::add
* prim::Constant
* aten::tanh
* aten::permute
* aten::cat

# Preparator Summary

yolov4_608_608_without_ABP_NMS Summary ───────────────────────────────────────╮
│ IR Version: 6                                                                                                        │
│ Opset Version: 11 ,                                                                                                  │
│ Producer Name:                                                                                                       │
│ Doc:                                                                                                                 │
│ Total Count of Ops: 424                                                                                              │
│ QModel Tool Version: 0.0.1                                                                                           │
│ Total Model Parameters: 64,329,953                                                                                   │
│                                                                                                                      │
│ All Ops: {'Conv': 110, 'Softplus': 72, 'Tanh': 72, 'Mul': 72, 'Add': 23, 'Concat': 12, 'LeakyRelu': 35, 'MaxPool':   │
│ 3, 'Shape': 2, 'Constant': 13, 'Slice': 2, 'Resize': 2, 'Reshape': 3, 'Transpose': 3}                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                         yolov4_608_608_without_ABP_NMS Detail                                          
╭───────────────────┬─────────────────────────────────────────────┬────────────────────────────────┬───────────────────╮
│ Name              │ Shape                                       │ Input/Output                   │ Dtype             │
├───────────────────┼─────────────────────────────────────────────┼────────────────────────────────┼───────────────────┤
│ images            │ [1, 3, 608, 608]                            │ input                          │ float32           │
│ 610               │ [1, 3, 76, 76, 85]                          │ output                         │ float32           │
│ 629               │ [1, 3, 38, 38, 85]                          │ output                         │ float32           │
│ 648               │ [1, 3, 19, 19, 85]                          │ output                         │ float32           │
╰───────────────────┴─────────────────────────────────────────────┴────────────────────────────────┴───────────────────╯
                                    Table Generated by QAicOnnxModel Preparator Tool                                    
2023-03-17 21:54:06.271 | INFO     | qaic_pytools.qmodel.preparator.preparator:summarize:150 - Final Summary of the Model
╭────────────────────────────── yolov4_608_608_without_ABP_NMS_preparator_aic100 Summary ──────────────────────────────╮
│ IR Version: 6                                                                                                        │
│ Opset Version: 11 , 3 ai.onnx.ml, 1 ai.onnx.training, 17 com.ms.internal.nhwc, 1 ai.onnx.preview.training, 1         │
│ com.microsoft, 1 com.microsoft.experimental, 1 com.microsoft.nchwc, 1 org.pytorch.aten,                              │
│ Producer Name:                                                                                                       │
│ Doc:                                                                                                                 │
│ Total Count of Ops: 408                                                                                              │
│ QModel Tool Version: 0.0.1                                                                                           │
│ Total Model Parameters: 64,329,972                                                                                   │
│                                                                                                                      │
│ All Ops: {'Conv': 110, 'Softplus': 72, 'Tanh': 72, 'Mul': 72, 'Add': 23, 'Concat': 10, 'LeakyRelu': 35, 'MaxPool':   │
│ 3, 'Resize': 2, 'Reshape': 3, 'Transpose': 3, 'Identity': 3}                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                yolov4_608_608_without_ABP_NMS_preparator_aic100 Detail                                 
╭────────────────────────────┬─────────────────────────────────────────────┬──────────────────────────┬────────────────╮
│ Name                       │ Shape                                       │ Input/Output             │ Dtype          │
├────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────┼────────────────┤
│ images                     │ ['batch', 3, 608, 608]                      │ input                    │ float32        │
│ feature_map_1              │ [1, 3, 76, 76, 85]                          │ output                   │ float32        │
│ feature_map_2              │ [1, 3, 38, 38, 85]                          │ output                   │ float32        │
│ feature_map_3              │ [1, 3, 19, 19, 85]                          │ output                   │ float32        │
╰────────────────────────────┴─────────────────────────────────────────────┴──────────────────────────┴────────────────╯
                                    Table Generated by QAicOnnxModel Preparator Tool                                    
╭───────────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────╮
│ Stages                                                                                        │ Status               │
├───────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
│ Native Checker                                                                                │ Passed               │
│ Internal(Ort) Checker                                                                         │ Passed               │
│ Shape Inference                                                                               │ Passed               │
│ Model Simplifier Optimization                                                                 │ Passed               │
│ Pattern Optimizer Required / Not Required                                                     │ Required             │
│ Dynamic model generation                                                                      │ Passed               │
│ Post Process Handler                                                                          │ Passed               │
│ Validation Checker                                                                            │ Passed               │
╰───────────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────╯
                                             Model Preparator Stage Status


## Open source ticket details
---

This model has been approved via ticket number [OSR Ticket](TBD) for
- [ ] can download
- [ ] can modify
- [ ] can use internally
- [ ] can redistribute modified files to customers with NDA/ some agreement
- [ ] can redistribute modified files to any one
- [ ] NO Open source approval sought


## References
---

* [YoloV4](https://github.com/ultralytics/yolov3/tree/archive)

## Point of contact
---

* Anuj Gupta : anujgupt@qti.qualcomm.com
* Vinayak N Baddi : vbaddi@qti.qualcomm.com
* Himanshu Upreti : hupreti@qti.qualcomm.com
* Prasanna Biswas : prasbisw@qti.qualcomm.com
* Palachandra M V : palachan@qti.qualcomm.com

## Legal
---

* [OSR Ticket](TBD)