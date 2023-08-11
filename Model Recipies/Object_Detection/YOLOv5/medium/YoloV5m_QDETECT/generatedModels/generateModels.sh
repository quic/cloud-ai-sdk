# Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
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

# Usage: sh generateModels.sh 5s 640 640

git clone https://github.com/ultralytics/yolov5.git

cd yolov5

git reset --hard c5ba2abb4afb9fe8c671f14eb5200647893efe30

git checkout

cd ..

pip install -r ./yolov5/requirements.txt

pip install -r ./requirements.txt

patch ./yolov5/models/yolo.py < yolo.patch 

patch ./yolov5/utils/downloads.py < downloads.patch 

mkdir ./ONNX

mkdir ./PyTorch



# For ONNX model with QNMS
python createModels.py --model_type $1 --op_folder ./ --model_with_qnms --iou_thresh 0.65 --score_thresh 0.001 --img_h $2 --img_w $3


rm -rf ./yolov5

rm -rf ./yolov$1.pt
