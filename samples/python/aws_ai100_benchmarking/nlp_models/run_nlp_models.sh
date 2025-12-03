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

mkdir -p ./bert-base-cased/
echo python run_nlp_model.py -m bert-base-cased -o best-latency "$@"
python run_nlp_model.py -m bert-base-cased -o best-latency "$@" | tee -a ./bert-base-cased/best-latency.log
echo python run_nlp_model.py -m bert-base-cased -o balanced "$@"
python run_nlp_model.py -m bert-base-cased -o balanced "$@" | tee -a ./bert-base-cased/balanced.log
echo python run_nlp_model.py -m bert-base-cased -o best-throughput "$@"
python run_nlp_model.py -m bert-base-cased -o best-throughput "$@" | tee -a ./bert-base-cased/best-throughput.log

mkdir -p ./bert-base-uncased/
echo python run_nlp_model.py -m bert-base-uncased -o best-latency "$@"
python run_nlp_model.py -m bert-base-uncased -o best-latency "$@" | tee -a ./bert-base-uncased/best-latency.log
echo python run_nlp_model.py -m bert-base-uncased -o balanced "$@"
python run_nlp_model.py -m bert-base-uncased -o balanced "$@" | tee -a ./bert-base-uncased/balanced.log
echo python run_nlp_model.py -m bert-base-uncased -o best-throughput "$@"
python run_nlp_model.py -m bert-base-uncased -o best-throughput "$@" | tee -a ./bert-base-uncased/best-throughput.log

mkdir -p ./bert-large-uncased/
echo python run_nlp_model.py -m bert-large-uncased -o best-latency "$@"
python run_nlp_model.py -m bert-large-uncased -o best-latency "$@" | tee -a ./bert-large-uncased/best-latency.log
echo python run_nlp_model.py -m bert-large-uncased -o balanced "$@"
python run_nlp_model.py -m bert-large-uncased -o balanced "$@" | tee -a ./bert-large-uncased/balanced.log
echo python run_nlp_model.py -m bert-large-uncased -o best-throughput "$@"
python run_nlp_model.py -m bert-large-uncased -o best-throughput "$@" | tee -a ./bert-large-uncased/best-throughput.log

mkdir -p ./distilbert-base-uncased/
echo python run_nlp_model.py -m distilbert-base-uncased -o best-latency "$@"
python run_nlp_model.py -m distilbert-base-uncased -o best-latency "$@" | tee -a ./distilbert-base-uncased/best-latency.log
echo python run_nlp_model.py -m distilbert-base-uncased -o balanced "$@"
python run_nlp_model.py -m distilbert-base-uncased -o balanced "$@" | tee -a ./distilbert-base-uncased/balanced.log
echo python run_nlp_model.py -m distilbert-base-uncased -o best-throughput "$@"
python run_nlp_model.py -m distilbert-base-uncased -o best-throughput "$@" | tee -a ./distilbert-base-uncased/best-throughput.log

mkdir -p ./google/electra-base-discriminator/
echo python run_nlp_model.py -m google/electra-base-discriminator -o best-latency "$@"
python run_nlp_model.py -m google/electra-base-discriminator -o best-latency "$@" | tee -a ./google/electra-base-discriminator/best-latency.log
echo python run_nlp_model.py -m google/electra-base-discriminator -o balanced "$@"
python run_nlp_model.py -m google/electra-base-discriminator -o balanced "$@" | tee -a ./google/electra-base-discriminator/balanced.log
echo python run_nlp_model.py -m google/electra-base-discriminator -o best-throughput "$@"
python run_nlp_model.py -m google/electra-base-discriminator -o best-throughput "$@" | tee -a ./google/electra-base-discriminator/best-throughput.log

mkdir -p ./Rostlab/prot_bert/
echo python run_nlp_model.py -m Rostlab/prot_bert -o best-latency "$@"
python run_nlp_model.py -m Rostlab/prot_bert -o best-latency "$@" | tee -a ./Rostlab/prot_bert/best-latency.log
echo python run_nlp_model.py -m Rostlab/prot_bert -o balanced "$@"
python run_nlp_model.py -m Rostlab/prot_bert -o balanced "$@" | tee -a ./Rostlab/prot_bert/balanced.log
echo python run_nlp_model.py -m Rostlab/prot_bert -o best-throughput "$@"
python run_nlp_model.py -m Rostlab/prot_bert -o best-throughput "$@" | tee -a ./Rostlab/prot_bert/best-throughput.log

mkdir -p ./roberta-base/
echo python run_nlp_model.py -m roberta-base -o best-latency "$@"
python run_nlp_model.py -m roberta-base -o best-latency "$@" | tee -a ./roberta-base/best-latency.log
echo python run_nlp_model.py -m roberta-base -o balanced "$@"
python run_nlp_model.py -m roberta-base -o balanced "$@" | tee -a ./roberta-base/balanced.log
echo python run_nlp_model.py -m roberta-base -o best-throughput "$@"
python run_nlp_model.py -m roberta-base -o best-throughput "$@" | tee -a ./roberta-base/best-throughput.log

mkdir -p ./roberta-large/
echo python run_nlp_model.py -m roberta-large -o best-latency "$@"
python run_nlp_model.py -m roberta-large -o best-latency "$@" | tee -a ./roberta-large/best-latency.log
echo python run_nlp_model.py -m roberta-large -o balanced "$@"
python run_nlp_model.py -m roberta-large -o balanced "$@" | tee -a ./roberta-large/balanced.log
echo python run_nlp_model.py -m roberta-large -o best-throughput "$@"
python run_nlp_model.py -m roberta-large -o best-throughput "$@" | tee -a ./roberta-large/best-throughput.log
