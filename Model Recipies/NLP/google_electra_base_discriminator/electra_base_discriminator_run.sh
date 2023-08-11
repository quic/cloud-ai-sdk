###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Export BERT Large to ONNX model
#       2. Compile and execute Mixed Precision and FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################


#Precision Options: 'int8'or 'fp16'
precision='int8'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=2

aic_binary_dir=./electra-base-discriminator-onnx-128

CONFIG_PATH=electra_base_discriminator.yaml


err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}

########################################################################
# Use Model Preparator tool to generate ONNX file optimized for AIC100.
# Globals:
#   Config File for Model Preparator
# Arguments:
#   None
# Outputs:
#   Generates Optimized ONNX File
########################################################################
prepare_model() {
	source /opt/qti-aic/dev/python/qaic-env/bin/activate
	if [[ -d ./WORKSPACE ]]; then
		sudo rm -ri ./WORKSPACE
	fi
	sudo python -W ignore /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config $CONFIG_PATH
}

###############################################################################
# Generate Random inputs as per the input requirements of the model generated.
# Globals:
#   None
# Arguments:
#   None
# Outputs:
#   Generates input files in "inputFiles" folder
###############################################################################
generate_inputs() {
	echo "In generate inputs"
        if [[ -d "./inputFiles" ]]; then
                rm -ri ./inputFiles/*
        fi
	mkdir inputFiles

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'int8' ]]; then
			python generate_inputs_onnx_model.py --model-path ./WORKSPACE/electra-base-discriminator_simplified_preparator_aic100.onnx --batch-size 1

                elif  [[ "${precision}" == 'fp16' ]]; then
			python generate_inputs_onnx_model.py --model-path ./WORKSPACE/electra-base-discriminator_simplified_preparator_aic100.onnx --batch-size 1
                fi


        elif [[ "${config}" -eq 2 ]]; then
		python generate_inputs_onnx_model.py --model-path ./WORKSPACE/electra-base-discriminator_simplified_preparator_aic100.onnx --batch-size 1
        fi
}


###############################################################################
# Compile ONNX file to generate QPC file which can be executed on AIC100.
# The generated QPC file shall give best Throughput or Latencny performance
# Globals:
#   aic_binary_dir
# Arguments:
#   None
# Outputs:
#   Generates QPC File in "aic_binary_dir" dir
###############################################################################
compile_model() {
        if [[ -d "${aic_binary_dir}" ]]; then
                rm -ri ${aic_binary_dir}
        fi

        compile_options='-aic-hw -aic-hw-version=2.0 -vvv -compile-only -m=./WORKSPACE/electra-base-discriminator_simplified_preparator_aic100.onnx -onnx-define-symbol=seq_length,128 -input-list-file=list.txt -aic-binary-dir='${aic_binary_dir}

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'int8' ]]; then
                        compile_options=${compile_options}" -mos=1 \
				 -aic-num-cores=2 \
				 -ols=1 \
				 -convert-to-fp16 \
				 -aic-num-of-instances=7 \
				 -onnx-define-symbol=batch_size,1 \
				 -onnx-define-symbol=sequence,128 \
                                 -stats-batchsize=1 \
                                 -multicast-weights  "
                fi
                if [[ "${precision}" == 'fp16' ]]
                then
                        compile_options=${compile_options}"  -convert-to-fp16 -mos=1 -ols=1 -aic-num-cores=4 -onnx-define-symbol=sequence,128 -onnx-define-symbol=batch_size,4 -stats-batchsize=4   -multicast-weights"

                fi
        elif [[ "${config}" -eq 2 ]]; then
                if [[ "${precision}" == 'int8' ]]; then
                        compile_options=${compile_options}" -mos=12 \
                                 -aic-num-cores=2 \
                                 -ols=1 \
                                 -convert-to-fp16 \
                                 -aic-num-of-instances=7 \
				 -onnx-define-symbol=batch_size,1 \
				 -onnx-define-symbol=sequence,128 \
                                 -stats-batchsize=1 \
                                 -multicast-weights  "
                fi
                if [[ "${precision}" == 'fp16' ]]; then
                        compile_options=${compile_options}"   -convert-to-fp16 -mos=14 -ols=1 -aic-num-cores=14  -onnx-define-symbol=batch_size,1 -onnx-define-symbol=sequence,128 -multicast-weights"

                fi
        fi

        /opt/qti-aic/exec/qaic-exec ${compile_options}

}

###############################################################################
# Execute the generated QPC file on AIC100.
# The Number of instances is configured using '-a' Flag
#
# Globals:
#   aic_binary_dir
# Arguments:
#   None
# Outputs:
#   Displays the performance metrics of the model like inf/sec etc.
###############################################################################
run_model() {

        runner_options=' --time 10  -d 0  -i ./inputFiles/input_ids.raw -i ./inputFiles/attention_mask.raw -i ./inputFiles/token_type_ids.raw -t '${aic_binary_dir}

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'mp' ]]; then
                        runner_options=${runner_options}' -a 3 '
                fi
                if [[ "${precision}" == 'fp16' ]]; then
                        runner_options=${runner_options}' -a 3 '

                fi
        elif [[ "${config}" -eq 2 ]]; then
                if [[ "${precision}" == 'mp' ]]; then
                        runner_options=${runner_options}' -a 1 '
                fi
                if [[ "${precision}" == 'fp16' ]]; then
                        runner_options=${runner_options}' -a 1 '
                fi


        fi
        sudo  sudo /opt/qti-aic/exec/qaic-runner ${runner_options}

}


if [[ -f ./electra-base-discriminator/generatedModels/electra-base-discriminator_simplified.onnx  ]]; then
	prepare_model
else
        err "BERT Base Model not found. Please make sure bert_setup.py is executed"
        exit 1
fi

generate_inputs
compile_model
run_model

