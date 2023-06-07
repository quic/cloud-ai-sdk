###############################################################################
#  Download the following model and other assets:
#    1) resnet50-v1-7.onnx: Resnet-50 model in ONNX format
#    2) cat_281_299.png: ImageNet test image
#    3) imagenet_class_index.json: ImageNet class list
###############################################################################

#!/bin/bash
# Remove if already exists.
if [ -e resnet50-v1-7.onnx ]; then
	rm -rf resnet50-v1-7.onnx
fi

if [ -e cat_281_299.png ]; then
	rm -rf cat_281_299.png
fi

if [ -e imagenet_class_index.json ]; then
	rm -rf imagenet_class_index.json
fi

# Download the following model and other assets
if [ ! -e resnet50-v1-7.onnx ]; then
    wget -c https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx
fi

image=cat_281_299.png

if [ ! -e ${image} ]; then
    wget -c https://github.com/pytorch/glow/raw/master/tests/images/imagenet_299/${image}
fi

if [ ! -e imagenet_class_index.json ]; then
    wget -c https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
fi
