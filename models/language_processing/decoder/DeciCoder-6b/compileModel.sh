#!/bin/bash

set -e

if [ -z "$1" ]; then
	echo "Usage: $0 <model_name>"
	exit 1
fi

model_name="$1"
batch_size="$2"
prompt_len="$3"
ctx_len="$4"
num_cores="$5"
with_or_no_mx="$6"

# Generate a new specializations.json
sed -e "s/BS/${batch_size}/g" -e "s/PL/${prompt_len}/g" -e "s/CL/${ctx_len}/g" ./specializations_template.json > specializations.json

# Create qpc directory - Delete exisiting path 
mkdir -p qpc
rm -rf qpc/${model_name}-kv-${prompt_len}pl-${ctx_len}cl-${num_cores}c${with_or_no_mx}

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

