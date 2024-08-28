#!/bin/bash

set -e

if [ -z "$1" ]; then
	echo "Usage: $0 <MODEL_NAME>"
	exit 1
fi

MODEL_NAME="$1"
BS="$2"
PL="$3"
CL="$4"
CORES="$5"
SOCS="$6"
FORMAT="$7"

python prepare_jsons.py --bs $BS --pl $PL --cl $CL --cores $CORES --socs $SOCS

if [ $FORMAT == "mx6" ]; then
    EXTRA="-mxfp6-matmul"
fi

if [ $SOCS != 1 ]; then
    EXTRA="-mdp-load-partition-config=mdp.json $EXTRA"
fi

# Create qpc directory
mkdir -p qpc
rm -rf qpc/${MODEL_NAME}-${BS}bs-${PL}pl-${CL}cl-$((CORES*SOCS))c-${FORMAT}

model_path="${MODEL_NAME}-kv/generatedModels/${MODEL_NAME}-kv_fp16_simplified.onnx"
if [ ! -f "$model_path" ]; then
	model_path="${MODEL_NAME}-kv/generatedModels/${MODEL_NAME}-kv.onnx"
fi

/opt/qti-aic/exec/qaic-exec \
	-m=$model_path \
	-aic-hw \
	-aic-hw-version=2.0 \
	-network-specialization-config=specializations.json \
	-retained-state \
	-convert-to-fp16 \
	-aic-num-cores=${CORES} \
	-custom-IO-list-file=${MODEL_NAME}-kv/custom_io.yaml \
	-compile-only \
	-aic-binary-dir=qpc/${MODEL_NAME}-${BS}bs-${PL}pl-${CL}cl-$((CORES*SOCS))c-${FORMAT} \
	${EXTRA}

