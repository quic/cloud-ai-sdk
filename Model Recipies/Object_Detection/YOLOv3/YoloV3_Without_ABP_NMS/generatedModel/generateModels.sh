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

# Usage: sh generateModels.sh 416 416

git clone https://github.com/ultralytics/yolov3 

cd yolov3

git checkout 1be31704c9c690929e4f6e6d950f40755ef2dcdc
# https://github.com/ultralytics/yolov3/commit/1be31704c9c690929e4f6e6d950f40755ef2dcdc -- Commit id used for cloning

cd ../

pip install -r ./yolov3/requirements.txt

pip install -r ./requirements.txt

patch ./yolov3/models/yolo.py < yolo-modified.patch 

patch ./yolov3/utils/google_utils.py < google_utils.patch 

mkdir ../modelFiles

python createModels.py --op_folder ../modelFiles/ --img_h $1 --img_w $2

rm -rf ./yolov3

rm -rf ./yolov3.pt
