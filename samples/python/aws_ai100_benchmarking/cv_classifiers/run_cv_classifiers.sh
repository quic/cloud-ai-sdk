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

mkdir -p ./resnet-152/
echo python run_cv_classifier.py -m resnet-152 -o best-latency "$@" 
python run_cv_classifier.py -m resnet-152 -o best-latency "$@" | tee -a ./resnet-152/best-latency.log
echo python run_cv_classifier.py -m resnet-152 -o balanced "$@" 
python run_cv_classifier.py -m resnet-152 -o balanced "$@" | tee -a ./resnet-152/balanced.log
echo python run_cv_classifier.py -m resnet-152 -o best-throughput "$@" 
python run_cv_classifier.py -m resnet-152 -o best-throughput "$@" | tee -a ./resnet-152/best-throughput.log

mkdir -p ./resnet-50/
echo python run_cv_classifier.py -m resnet-50 -o best-latency "$@" 
python run_cv_classifier.py -m resnet-50 -o best-latency "$@" | tee -a ./resnet-50/best-latency.log
echo python run_cv_classifier.py -m resnet-50 -o balanced "$@" 
python run_cv_classifier.py -m resnet-50 -o balanced "$@" | tee -a ./resnet-50/balanced.log
echo python run_cv_classifier.py -m resnet-50 -o best-throughput "$@" 
python run_cv_classifier.py -m resnet-50 -o best-throughput "$@" | tee -a ./resnet-50/best-throughput.log

mkdir -p ./vit-base-patch16-224/
echo python run_cv_classifier.py -m vit-base-patch16-224 -o best-latency "$@" 
python run_cv_classifier.py -m vit-base-patch16-224 -o best-latency "$@" | tee -a ./vit-base-patch16-224/best-latency.log
echo python run_cv_classifier.py -m vit-base-patch16-224 -o balanced "$@" 
python run_cv_classifier.py -m vit-base-patch16-224 -o balanced "$@" | tee -a ./vit-base-patch16-224/balanced.log
echo python run_cv_classifier.py -m vit-base-patch16-224 -o best-throughput "$@" 
python run_cv_classifier.py -m vit-base-patch16-224 -o best-throughput "$@" | tee -a ./vit-base-patch16-224/best-throughput.log
