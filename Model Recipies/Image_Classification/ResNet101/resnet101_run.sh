###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Use Model Preparator tool for generating ONNX model optimized for AIC100 
#       2. Compile and execute int8 and FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################



#Precision Options: 'int8'or 'fp16'
precision='int8'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=2

CONFIG_FILE=./resnetv1_model_info.yaml
aic_binary_dir=./resnet101-onnx/


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

        if [[ -d "WORKSPACE" ]]; then
                rm -ri "WORKSPACE"
        fi

	source /opt/qti-aic/dev/python/qaic-env/bin/activate
	sudo python -W ignore /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config $CONFIG_FILE
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
generate_inputs(){
        if [[ -f "inputFiles/input.1.raw" ]]; then
                rm inputFiles/input.1.raw
        fi

	
	if [ ${config} -eq 1 ]
        then
                if [ ${precision} == 'fp16' ]
                then
		        python generate_inputs_onnx_model.py --model-path ./WORKSPACE_RESNET101/ResNet101_preparator_aic100.onnx --batch-size 8
                fi

                if [ ${precision} == 'int8' ]
                then
		        python generate_inputs_onnx_model.py --model-path ./WORKSPACE_RESNET101/ResNet101_preparator_aic100.onnx --batch-size 16
                fi

        elif [ ${config} -eq 2 ]
        then
		        python generate_inputs_onnx_model.py --model-path ./WORKSPACE_RESNET101/ResNet101_preparator_aic100.onnx --batch-size 1
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
	if [ -d ${aic_binary_dir} ]
	then
		rm -ri ${aic_binary_dir}
	fi

	        compile_options='-aic-hw -aic-hw-version=2.0 -vvv -compile-only -m=./WORKSPACE_RESNET101/ResNet101_preparator_aic100.onnx -aic-binary-dir='${aic_binary_dir}

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'fp16' ]]; then
                        compile_options=${compile_options}" -convert-to-fp16 -onnx-define-symbol=batch_size,8 -stats-batchsize=8 -mos=1 -ols=2 -aic-num-cores=4  -input-list-file=./list.txt -multicast-weights  "
                fi

                if [[ "${precision}" == 'int8' ]]
                then
                        compile_options=${compile_options}"   -convert-to-quantize -quantization-schema=symmetric_with_uint8 -quantization-precision=Int8  -onnx-define-symbol=batch_size,16 -stats-batchsize=16 -mos=1 -ols=4 -aic-num-cores=4 -input-list-file=./list.txt   -multicast-weights"
                fi

        elif [[ "${config}" -eq 2 ]]; then
                if  [[ "${precision}" == 'fp16' ]]; then
                        compile_options=${compile_options}"  -convert-to-fp16  -onnx-define-symbol=batch_size,1 -stats-batchsize=1 -mos=6 -ols=1 -aic-num-cores=6 -input-list-file=./list.txt -multicast-weights  "
                fi
                if [[ "${precision}" == 'int8' ]]; then
                         compile_options=$compile_options" -convert-to-quantize -quantization-schema=symmetric_with_uint8 -quantization-precision=Int8  -onnx-define-symbol=batch_size,1 -stats-batchsize=1 -mos=6 -ols=1 -aic-num-cores=6  -input-list-file=./list.txt  -multicast-weights  "
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
	        runner_options=' --time 10  -d 1 -i ./inputFiles/input.1.raw  -t '${aic_binary_dir}

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'fp16' ]]; then
                        runner_options=${runner_options}' -a 3'
                fi

                if [[ "${precision}" == 'int8' ]]; then
                        runner_options=${runner_options}' -a 3'
                fi
        elif [[ "${config}" -eq 2 ]]
        then
                if  [[ "${precision}" == 'fp16' ]]; then
                        runner_options=${runner_options}' -a 2 -S 1'
                fi
                if [[ "${precision}" == 'int8' ]]; then
                        runner_options=${runner_options}' -a 2 -S 1'
                fi
        fi


        sudo /opt/qti-aic/exec/qaic-runner ${runner_options}
	deactivate

	}

if [[ -f "${CONFIG_FILE}" ]]; then
	prepare_model
else
	err "Config File for Model Preparator does not exist"
	exit 1
fi

if [[ -f "./WORKSPACE_RESNET101/ResNet101_preparator_aic100.onnx" ]]; then
	generate_inputs
	compile_model
else
        err "ONNX File not generated. Is QAIC Pytools installed?"
        exit 1
fi

run_model
