#!/bin/bash

##############################################################################
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
##############################################################################

if [ -z "$1" ]; then
	echo "Usage: $0 --model-repo <model_repo> --prompt \"<input prompt>\" --tlm-model-name <tlm_model_name> --tlm-precision [mx6 | fp16] --dlm-precision [mx6 | fp16] --dlm-model-name <dlm_model_name> --spec-length <DLM speculation length> --num-cores-dlm <0-16> --num-cores-tlm <0-16> \
  --pl <prompt length> --cl <context length> [--exact-greedy] [--mq] [--device-ids <comma-separated-device-ids>]"
	exit 1
fi

model_repo=
tlm_model_name=
dlm_model_name=
model_family=
tlm_precision=
dlm_precision=
pl=
cl=
ncores_tlm=
ncores_dlm=
spec_length=
exact_greedy=
device_id=
prompt_str=
qpc_dir="./qpc"

log_out_dir="./output_logs"
if ! test -d $log_out_dir; then
  mkdir $log_out_dir
fi

while [ $# -gt 0 ]; do
	case "$1" in
		--tlm-model-name)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			tlm_model_name=$2
			shift
			;;
		--dlm-model-name)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			dlm_model_name=$2
			shift
			;;
		--tlm-precision)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			tlm_precision=$2
			shift
			;;
		--dlm-precision)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			dlm_precision=$2
			shift
			;;
		--model-repo)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			model_repo=$2
			shift
			;;
		--model-family)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			model_family=$2
			shift
			;;
		--spec-length)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			spec_length=$2
			shift
			;;
		--device-ids)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			device_id=$2
			shift
			;;
		--num-cores-tlm)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			ncores_tlm=$2
			shift
			;;
		--num-cores-dlm)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			ncores_dlm=$2
			shift
			;;
		--pl)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			pl=$2
			shift
			;;
		--cl)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			cl=$2
			shift
			;;
		--mq)
			qpc_dir="./qpc_mq"
			shift
			;;
		--prompt)
			if [ $# -eq 1 ]; then echo "Error: missing $1 argument"; exit 1; fi
			prompt_str=$2
			shift
			;;
		--exact-greedy)
			exact_greedy="--exact-greedy"
			shift
			;;
		*)
			shift
			;;
	esac
done

if [ "z${tlm_precision}" == "z" ]; then tlm_precision="fp16"; fi
if [ "z${dlm_precision}" == "z" ]; then dlm_precision="fp16"; fi
if [ "z${prompt_str}" == "z" ]; then echo "Error: missing --prompt argument"; exit 1; fi
if [ "z${tlm_model_name}" == "z" ]; then echo "Error: missing --tlm-model-name argument"; exit 1; fi
if [ "z${dlm_model_name}" == "z" ]; then echo "Error: missing --dlm-model-name argument"; exit 1; fi
if [ "z${model_repo}" == "z" ]; then echo "Error: missing --model-repo argument"; exit 1; fi
if [ "z${model_family}" == "z" ]; then echo "Error: missing --model-family argument"; exit 1; fi
if [ "z${spec_length}" == "z" ]; then echo "Error: missing --spec-length argument"; exit 1; fi
if [ "z${ncores_tlm}" == "z" ]; then echo "Error: missing --num-cores-tlm argument"; exit 1; fi
if [ "z${ncores_dlm}" == "z" ]; then echo "Error: missing --num-cores-dlm argument"; exit 1; fi
if [ "z${pl}" == "z" ]; then echo "Error: missing --pl argument"; exit 1; fi
if [ "z${cl}" == "z" ]; then echo "Error: missing --cl argument"; exit 1; fi

if ! test -d $qpc_dir; then echo "Error: missing QPC directory: ${qpc_dir}"; exit 1; fi

full_model_path="${model_repo}/${tlm_model_name}"

execute_run_cmd () {
python ./speculativeDecoding.py $exact_greedy --model-family ${model_family} --model-name ${full_model_path} --tlm-qpc \
${qpc_dir}/${tlm_model_name}-kv-${pl}pl-${cl}cl-${ncores_tlm}c-_${spec_length}speclen-${tlm_precision}-tlm --dlm-qpc \
${qpc_dir}/${dlm_model_name}-kv-${pl}pl-${cl}cl-${ncores_dlm}c-speclen-${dlm_precision}-dlm --prompt "${prompt_str}" \
--prompt-len ${pl} --ctx-len ${cl} --max-spec-length ${spec_length} --device_id ${device_id}
}

execute_run_cmd
