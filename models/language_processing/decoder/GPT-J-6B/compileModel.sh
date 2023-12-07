#!/bin/bash

set -e

# Create qpc directory
mkdir -p qpc

num_cores=14

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
	echo "Usage: $0 <model_name> <mx6, fp16> <num_nsp>"
	exit 1
fi

if
	[ "$2" != "mx6" ] && [ $2 != "fp16" ]; then
	echo "Usage: $0 <model_name> <mx6, fp16> <num_nsp>"
	echo "Precision must be either mx6 or fp16"
	exit 1
fi

if (( $3 < 1 || $3 > 14 )); then
	echo "Usage: $0 <model_name> <mx6, fp16> <num_nsp>"
	echo "num_nsp must be integer in range 1-14"
	exit 1
fi

model_name="$1"
prec="$2"
num_cores="$3"
prompt_len=$(grep seq_len specializations.json | head -n1 | grep -Eo '[[:digit:]]+')
ctx_len=$(grep ctx_len specializations.json | head -n1 | grep -Eo '[[:digit:]]+')
num_blocks=$(grep 'value.' ${model_name}-kv/custom_io.yaml | tail -n1 | grep -Eo '[[:digit:]]+')


model_path="${model_name}-kv/generatedModels/${model_name}-kv_simplified.onnx"
if [ ! -f "$model_path" ]; then
	model_path="${model_name}-kv/generatedModels/${model_name}-kv.onnx"
fi

if [ "$2" = "mx6" ]; then
	prec="-mxfp6-matmul"
else
	prec=""
fi


/opt/qti-aic/exec/qaic-exec \
	-m=${model_name}-kv/generatedModels/${model_name}-kv_simplified.onnx \
	-aic-hw \
	-aic-hw-version=2.0 \
	-network-specialization-config=specializations.json \
	-convert-to-fp16 \
    ${prec} \
	-retained-state=true \
	-aic-num-cores=${num_cores} \
	-custom-IO-list-file=${model_name}-kv/custom_io.yaml  \
	-compile-only \
	-aic-binary-dir=qpc/${model_name}-kv-${prompt_len}pl-${ctx_len}cl-${num_cores}c
