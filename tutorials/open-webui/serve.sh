#!/bin/bash

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

model=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4

/opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model ${model} --max-model-len 4096 --max-num-seq 1 --max-seq_len-to-capture 128 --device qaic --device-group 0,1,2,3
