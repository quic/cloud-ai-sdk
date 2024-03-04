#!/bin/bash

##############################################################################
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
##############################################################################

set -e

if [ -z "$1" ]; then
	echo "Usage: $0 --model-name <model_name> --spec-length <spec_len> --<tlm | dlm> --num-cores <0-16> [--num-devices <number of devices>]"
	exit 1
fi

mx6orfp16=
num_devices=1
mdp_spec_fname_arg=
num_cores_arg=
qpc_dirname=
spec_length=
while [ $# -gt 0 ]; do
	case "$1" in
	  --model-name)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			model_name=$2
			shift
			;;
		--spec-length)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			spec_length=$2
			shift
			;;
		--mx6)
			if [ "$mx6orfp16" == "fp16" ]; then echo "Error: don't provide both --mx6 and --fp16 at once"; exit 1; fi
			mx6orfp16="mx6"
			shift
			;;
		--fp16)
			if [ "$mx6orfp16" == "mx6" ]; then echo "Error: don't provide both --mx6 and --fp16 at once"; exit 1; fi
			mx6orfp16="fp16"
			shift
			;;
		--num-cores)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			num_cores=$2
			shift
			;;
		--num-devices)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			num_devices=$2
			shift
			;;
		--tlm)
			if [ "$tlm_or_dlm" == "dlm" ]; then echo "Error: don't provide both --tlm and --dlm at once"; exit 1; fi
			tlm_or_dlm="tlm"
			shift
			;;
		--dlm)
			if [ "$tlm_or_dlm" == "tlm" ]; then echo "Error: don't provide both --tlm and --dlm at once"; exit 1; fi
			tlm_or_dlm="dlm"
			shift
			;;
		*)
			shift
			;;
	esac
done
if ["z${mx6orfp16}" == "z" ]; then mx6orfp16="fp16"; fi
if [ "z${model_name}" == "z" ]; then echo "Error: missing --model-name argument"; exit 1; fi
if [ "z${tlm_or_dlm}" == "z" ]; then echo "Error: missing both --tlm and --dlm argument"; exit 1; fi
if [ "z${num_cores}" == "z" ]; then echo "Error: missing --num-cores argument"; exit 1; fi
if [ "${tlm_or_dlm}" == "tlm" ]; then
		if [ "z${spec_length}" == "z" ]; then echo "Error: missing --spec-length argument"; exit 1; fi
fi
if [ ${num_devices} -gt 1 ]; then
		mdp_spec_fname="./multi-device-config/multi_device_partitioning_${num_devices}x${num_cores}.json"
		if ! test -f ${mdp_spec_fname}; then echo "Error: MDP config file ${mdp_spec_fname} not found"; exit 1; fi
		mdp_spec_fname_arg=" -mdp-load-partition-config=${mdp_spec_fname}"
		qpc_dir="./qpc_mq"
else
		#provide num cores as qaic-exec argument only when non-mq (for asymmetric num_cores assignment support)
		num_cores_arg=" -aic-num-cores=${num_cores}"
		qpc_dir="./qpc"
fi
spec_len_suffix="_${spec_length}"
if [ "$tlm_or_dlm" == "dlm" ]; then
	spec_len_suffix=""
fi
net_spec_fname="network-specializations/${tlm_or_dlm}_specializations${spec_len_suffix}.json"
prompt_len=$(grep seq_len ${net_spec_fname} | head -n1 | grep -Eo '[[:digit:]]+')
ctx_len=$(grep ctx_len ${net_spec_fname} | head -n1 | grep -Eo '[[:digit:]]+')
num_blocks=$(grep 'value.' ${model_name}-kv/custom_io.yaml | tail -n1 | grep -Eo '[[:digit:]]+')

# Create qpc directory
mkdir -p ${qpc_dir}

model_path="${model_name}-kv/generatedModels/${model_name}-kv_fp16_simplified.onnx"
if [ ! -f "$model_path" ]; then
	model_path="${model_name}-kv/generatedModels/${model_name}-kv_fp16.onnx"
fi

runcmd="/opt/qti-aic/exec/qaic-exec \
	-m=${model_path} \
	-aic-hw \
	-aic-hw-version=2.0 \
	-network-specialization-config=${net_spec_fname} \
	-retained-state \
	-convert-to-fp16 \
	-custom-IO-list-file=${model_name}-kv/custom_io.yaml \
	-compile-only \
	-aic-binary-dir=${qpc_dir}/${model_name}-kv-${prompt_len}pl-${ctx_len}cl-${num_cores}c-${spec_len_suffix}speclen-${mx6orfp16}-${tlm_or_dlm} \
	${num_cores_arg} ${mdp_spec_fname_arg}"
if [ "${mx6orfp16}" == "mx6" ]; then
    runcmd+=" -mxfp6-matmul"
fi

echo "$runcmd"
$runcmd
