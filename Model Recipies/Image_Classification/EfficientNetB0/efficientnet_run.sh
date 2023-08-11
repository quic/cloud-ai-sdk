###########################################################################################################################
#!/bin/bash
#
#This script compiles and executes int8 and FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################


#Precision Options: 'int8'or 'fp16'
precision='fp16'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1


aic_binary_dir=./EfficientnetB0-onnx/

err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
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
        if [[ -d ./inputFiles ]]; then
                rm -ri ./inputFiles
        fi
	python generate_inputs_onnx_model.py --model-path  efficientnetb0.onnx  --batch-size 1 

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
        compile_options='-aic-hw -aic-hw-version=2.0 -vvv -compile-only -m=efficientnetb0.onnx -input-list-file=file-list.txt -aic-binary-dir='${aic_binary_dir}

        if [[ "${config}" -eq 1 ]]
        then
                if [[ "${precision}" == 'int8' ]]; then
                        compile_options=${compile_options}" -convert-to-quantize -quantization-schema=symmetric_with_uint8 -quantization-precision=Int8 -batchsize=1 -mos=1 -ols=1 -aic-num-cores=1 -multicast-weights "
                elif [[ "${precision}" == 'fp16' ]]; then
                         compile_options=${compile_options}" -convert-to-fp16 -batchsize=1 -mos=1 -ols=1 -aic-num-cores=1 -multicast-weights "
                fi

        elif [[ "${config}" -eq 2 ]]; then
                if [[ "${precision}" == 'int8' ]]; then
                        compile_options=${compile_options}"   -convert-to-quantize -quantization-schema=symmetric_with_uint8 -quantization-precision=Int8 -batchsize=1 -mos=4 -ols=1 -aic-num-cores=4 -multicast-weights  "
                elif [[ "${precision}" == 'fp16' ]]; then
                         compile_options=${compile_options}"  -convert-to-fp16 -batchsize=1 -mos=4 -ols=1 -aic-num-cores=4 -multicast-weights "
                fi

        fi

        /opt/qti-aic/exec/qaic-exec "${compile_options}"

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
        runner_options=' --time 10  -d 1   -i ./inputFiles/inp.raw  -t '${aic_binary_dir}
        if [[ "${config}" -eq 2 ]]; then
                runner_options=${runner_options}' -a 3 -S 1'
        elif [[ "${config}" -eq 1 ]]; then
                runner_options=${runner_options}' -a 14 '
        fi

        sudo  sudo /opt/qti-aic/exec/qaic-runner "${runner_options}"

        }

if [[ -f "./WORKSPACE_RESNET101/ResNet101_preparator_aic100.onnx" ]]; then
	generate_inputs
else
        err "ONNX File not found. Pls run efficientnet_setup.sh"
        exit 1
fi

compile_model
run_model
