##############################################################################
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023, Qualcomm Technologies, Inc. All Rights Reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
##############################################################################

#!/bin/bash

mkdir -p ./yolov4/
echo python run_yolo_model.py -m yolov4 -o best-latency "$@" 
python run_yolo_model.py -m yolov4 -o best-latency "$@" | tee -a ./yolov4/best-latency.log
echo python run_yolo_model.py -m yolov4 -o balanced "$@" 
python run_yolo_model.py -m yolov4 -o balanced "$@" | tee -a ./yolov4/balanced.log
echo python run_yolo_model.py -m yolov4 -o best-throughput "$@" 
python run_yolo_model.py -m yolov4 -o best-throughput "$@" | tee -a ./yolov4/best-throughput.log

echo python run_yolo_model.py --include-nms -m yolov4 -o best-latency "$@" 
python run_yolo_model.py --include-nms -m yolov4 -o best-latency "$@" | tee -a ./yolov4/best-latency-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov4 -o balanced "$@" 
python run_yolo_model.py --include-nms -m yolov4 -o balanced "$@" | tee -a ./yolov4/balanced-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov4 -o best-throughput "$@" 
python run_yolo_model.py --include-nms -m yolov4 -o best-throughput "$@" | tee -a ./yolov4/best-throughput-with-nms.log


mkdir -p ./yolov5s/
echo python run_yolo_model.py -m yolov5s -o best-latency "$@" 
python run_yolo_model.py -m yolov5s -o best-latency "$@" | tee -a ./yolov5s/best-latency.log
echo python run_yolo_model.py -m yolov5s -o balanced "$@" 
python run_yolo_model.py -m yolov5s -o balanced "$@" | tee -a ./yolov5s/balanced.log
echo python run_yolo_model.py -m yolov5s -o best-throughput "$@" 
python run_yolo_model.py -m yolov5s -o best-throughput "$@" | tee -a ./yolov5s/best-throughput.log

echo python run_yolo_model.py --include-nms -m yolov5s -o best-latency "$@" 
python run_yolo_model.py --include-nms -m yolov5s -o best-latency "$@" | tee -a ./yolov5s/best-latency-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov5s -o balanced "$@" 
python run_yolo_model.py --include-nms -m yolov5s -o balanced "$@" | tee -a ./yolov5s/balanced-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov5s -o best-throughput "$@" 
python run_yolo_model.py --include-nms -m yolov5s -o best-throughput "$@" | tee -a ./yolov5s/best-throughput-with-nms.log


mkdir -p ./yolov5m/
echo python run_yolo_model.py -m yolov5m -o best-latency "$@" 
python run_yolo_model.py -m yolov5m -o best-latency "$@" | tee -a ./yolov5m/best-latency.log
echo python run_yolo_model.py -m yolov5m -o balanced "$@" 
python run_yolo_model.py -m yolov5m -o balanced "$@" | tee -a ./yolov5m/balanced.log
echo python run_yolo_model.py -m yolov5m -o best-throughput "$@" 
python run_yolo_model.py -m yolov5m -o best-throughput "$@" | tee -a ./yolov5m/best-throughput.log

echo python run_yolo_model.py --include-nms -m yolov5m -o best-latency "$@" 
python run_yolo_model.py --include-nms -m yolov5m -o best-latency "$@" | tee -a ./yolov5m/best-latency-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov5m -o balanced "$@" 
python run_yolo_model.py --include-nms -m yolov5m -o balanced "$@" | tee -a ./yolov5m/balanced-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov5m -o best-throughput "$@" 
python run_yolo_model.py --include-nms -m yolov5m -o best-throughput "$@" | tee -a ./yolov5m/best-throughput-with-nms.log


mkdir -p ./yolov5l/
echo python run_yolo_model.py -m yolov5l -o best-latency "$@" 
python run_yolo_model.py -m yolov5l -o best-latency "$@" | tee -a ./yolov5l/best-latency.log
echo python run_yolo_model.py -m yolov5l -o balanced "$@" 
python run_yolo_model.py -m yolov5l -o balanced "$@" | tee -a ./yolov5l/balanced.log
echo python run_yolo_model.py -m yolov5l -o best-throughput "$@" 
python run_yolo_model.py -m yolov5l -o best-throughput "$@" | tee -a ./yolov5l/best-throughput.log

echo python run_yolo_model.py --include-nms -m yolov5l -o best-latency "$@" 
python run_yolo_model.py --include-nms -m yolov5l -o best-latency "$@" | tee -a ./yolov5l/best-latency-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov5l -o balanced "$@" 
python run_yolo_model.py --include-nms -m yolov5l -o balanced "$@" | tee -a ./yolov5l/balanced-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov5l -o best-throughput "$@" 
python run_yolo_model.py --include-nms -m yolov5l -o best-throughput "$@" | tee -a ./yolov5l/best-throughput-with-nms.log


mkdir -p ./yolov5x/
echo python run_yolo_model.py -m yolov5x -o best-latency "$@" 
python run_yolo_model.py -m yolov5x -o best-latency "$@" | tee -a ./yolov5x/best-latency.log
echo python run_yolo_model.py -m yolov5x -o balanced "$@" 
python run_yolo_model.py -m yolov5x -o balanced "$@" | tee -a ./yolov5x/balanced.log
echo python run_yolo_model.py -m yolov5x -o best-throughput "$@" 
python run_yolo_model.py -m yolov5x -o best-throughput "$@" | tee -a ./yolov5x/best-throughput.log

echo python run_yolo_model.py --include-nms -m yolov5x -o best-latency "$@" 
python run_yolo_model.py --include-nms -m yolov5x -o best-latency "$@" | tee -a ./yolov5x/best-latency-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov5x -o balanced "$@" 
python run_yolo_model.py --include-nms -m yolov5x -o balanced "$@" | tee -a ./yolov5x/balanced-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov5x -o best-throughput "$@" 
python run_yolo_model.py --include-nms -m yolov5x -o best-throughput "$@" | tee -a ./yolov5x/best-throughput-with-nms.log


mkdir -p ./yolov7-e6e/
echo python run_yolo_model.py -m yolov7-e6e -o best-latency "$@" 
python run_yolo_model.py -m yolov7-e6e -o best-latency "$@" | tee -a ./yolov7-e6e/best-latency.log
echo python run_yolo_model.py -m yolov7-e6e -o balanced "$@" 
python run_yolo_model.py -m yolov7-e6e -o balanced "$@" | tee -a ./yolov7-e6e/balanced.log
echo python run_yolo_model.py -m yolov7-e6e -o best-throughput "$@" 
python run_yolo_model.py -m yolov7-e6e -o best-throughput "$@" | tee -a ./yolov7-e6e/best-throughput.log

echo python run_yolo_model.py --include-nms -m yolov7-e6e -o best-latency "$@" 
python run_yolo_model.py --include-nms -m yolov7-e6e -o best-latency "$@" | tee -a ./yolov7-e6e/best-latency-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov7-e6e -o balanced "$@" 
python run_yolo_model.py --include-nms -m yolov7-e6e -o balanced "$@" | tee -a ./yolov7-e6e/balanced-with-nms.log
echo python run_yolo_model.py --include-nms -m yolov7-e6e -o best-throughput "$@" 
python run_yolo_model.py --include-nms -m yolov7-e6e -o best-throughput "$@" | tee -a ./yolov7-e6e/best-throughput-with-nms.log

