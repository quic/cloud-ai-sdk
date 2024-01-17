#!/bin/bash

set -e

if [ -z "$1" ]; then
	echo "Usage: $0 <model_name>"
	exit 1
fi

model_name="$1"
prompt_len=$(grep seq_len specializations.json | head -n1 | grep -Eo '[[:digit:]]+')
ctx_len=$(grep ctx_len specializations.json | head -n1 | grep -Eo '[[:digit:]]+')
num_cores="$2"
num_blocks=$(grep 'value.' ${model_name}-kv/custom_io.yaml | tail -n1 | grep -Eo '[[:digit:]]+')
with_or_no_mx="$3"



# Create qpc directory
mkdir -p qpc

model_path="${model_name}-kv/generatedModels/${model_name}-kv_fp16_simplified.onnx"
if [ ! -f "$model_path" ]; then
	model_path="${model_name}-kv/generatedModels/${model_name}-kv_fp16.onnx"
fi

/opt/qti-aic/exec/qaic-exec \
	-m=$model_path \
	-aic-hw \
	-aic-hw-version=2.0 \
	-network-specialization-config=specializations.json \
	-retained-state \
	-convert-to-fp16 \
	-ols=1 \
	-mos=${num_cores} \
	-aic-num-cores=${num_cores} \
	-custom-IO-list-file=${model_name}-kv/custom_io.yaml \
	-compile-only \
	-aic-binary-dir=qpc/${model_name}-kv-${prompt_len}pl-${ctx_len}cl-${num_cores}c${with_or_no_mx} \
	${with_or_no_mx}

