#!/bin/bash

set -e

if [ -z "$1" ]; then
	echo "Usage: $0 <model_name>"
	exit 1
fi

model_name="$1"
BS="$2"
PL="$3"
CL="$4"
CORES="$5"
MOS="$6"
OLS="$7"
SOCS="$8"
MX="$9"

sed -e "s/CORES/${CORES}/g" ./mdp_template.json  >  mdp.json
sed -e "s/BS/${BS}/g" -e "s/PL/${PL}/g" -e "s/CL/${CL}/g" ./specializations_template.json  >  specializations.json	
if [ $PL == 1 ]; then
    sed -e "s/BS/${BS}/g" -e "s/PL/${PL}/g" -e "s/CL/${CL}/g" ./specializations_template_batch.json  >  specializations.json
fi	

if [ $MX != "-mxfp6-matmul" ]; then
    MX=""
fi

MDP=""
if [ $SOCS == 4 ]; then
    MDP="-mdp-load-partition-config=mdp.json "
fi

EXTRA="$MDP$MX"

# Create qpc directory
mkdir -p qpc
rm -rf qpc/${model_name}-${PL}pl-${CL}cl-${CORES}c${MX}

model_path="${model_name}/generatedModels/${model_name}_fp16_simplified.onnx"
if [ ! -f "$model_path" ]; then
	model_path="${model_name}/generatedModels/${model_name}.onnx"
fi


/opt/qti-aic/exec/qaic-exec \
	-m=$model_path \
	-aic-hw \
	-aic-hw-version=2.0 \
	-network-specialization-config=specializations.json \
	-retained-state \
	-convert-to-fp16 \
	-ols=${OLS} \
	-mos=${MOS} \
	-aic-num-cores=${CORES} \
	-custom-IO-list-file=${model_name}/custom_io.yaml \
	-compile-only \
	-aic-binary-dir=qpc/${model_name}-${PL}pl-${CL}cl-${CORES}c${MX} \
	${EXTRA}

