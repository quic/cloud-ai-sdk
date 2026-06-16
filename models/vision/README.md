## Description

This directory contains computer vision models for Qualcomm Cloud AI accelerators. These models cover a wide range of vision tasks including image classification, object detection and instance segmentation.

## Model Categories

### [Classification](classification/)

Image classification models that assign labels to entire images. Includes:

- **ViT** (Vision Transformer): `vit_b_16`, `vit_b_32`, `vit-base-patch16-224`
- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **ResNeXt**: `resnext101_32x8d`, `resnext101_64x4d`, `resnext50_32x4d`
- **Wide ResNet**: `wide_resnet101_2`, `wide_resnet50_2`
- **DenseNet**: `densenet121`, `densenet161`, `densenet169`, `densenet201`
- **MobileNet**: `mobilenet_v2`, `mobilenet_v3_large`, `mobilenet_v3_small`
- **EfficientNet**: `efficientnet_v2_l`, `efficientnet_v2_m`, `efficientnet_v2_s`, `efficientnet_b0`, `efficientnet_b7`
- **MNASNet**: `mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, `mnasnet1_3`
- **ShuffleNet**: `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`
- **SqueezeNet**: `squeezenet1_0`, `squeezenet1_1`
- **VGG**: `vgg11`, `vgg13`, `vgg16`, `vgg19` (with and without BN)
- **Inception**: `inception_v3`
- **AlexNet**: `alexnet`

### [Detection](detection/)

Object detection models that locate and classify objects with bounding boxes:

- **YOLOv5**: `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`
- **YOLOv7**: `yolov7-e6e`
- **YOLOv8**: `yolov8m`

Includes Docker Compose deployment with Triton server and inference clients.

### [Segmentation](segmentation/)

Instance segmentation models that provide pixel-level object masks:

- **Mask R-CNN**: Detectron2-based implementation with bounding boxes and masks

Includes Triton server deployment and demo utilities.

## Resources

- See individual subdirectory READMEs for model-specific documentation
- Refer to the main [README](../../..) for SDK overview
- Check [Tutorials](../../tutorials) for usage examples
