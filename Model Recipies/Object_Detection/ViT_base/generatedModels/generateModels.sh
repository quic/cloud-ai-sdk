#!/usr/bin/env python3
#
# Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.

mkdir ../inputFiles

echo "Downloading imagenet label mapping file"
wget -O ./../inputFiles/imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

echo "Downloading sample image file for inference"
wget -O ./../inputFiles/000000039769.jpg http://images.cocodataset.org/val2017/000000039769.jpg

echo "Generating model with 224x224 input"
python convertModel.py 224


