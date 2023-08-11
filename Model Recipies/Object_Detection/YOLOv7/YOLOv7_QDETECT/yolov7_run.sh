###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Export BERT Large to ONNX model 
#       2. Compile and execute Mixed Precision and FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################


#Precision Options: 'int8'or 'fp16'
precision='fp16'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

aic_binary_dir=./qpc_dir


err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}

########################################################################
# Generate ONNX file for BERT Large
# Arguments:
#   None
# Outputs:
#   Generates Optimized ONNX File
########################################################################
prepare_model() {
	source /opt/qti-aic/dev/python/qaic-env/bin/activate
	python /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config yolov7_qdetect.yaml
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
	if [[ -d "./inputFiles" ]]; then
                rm -ri ./inputFiles/*.raw
        fi
	python generateRawYolov7.py --img_path inputFiles/horses.jpg --h_w 640 640
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

        compile_options='-aic-hw -aic-hw-version=2.0 -vvv -input-list-file=./input.txt -compile-only -m=./WORKSPACE/yolov7_640_640_smartNMS_preparator_aic100.onnx -aic-binary-dir='${aic_binary_dir}



        if [[ "${config}" -eq 1 ]]; then
		if [[ "${precision}" == 'int8' ]]; then
			compile_options=${compile_options}" -mos=1 \
				-ols=1 \
				-convert-to-quantize \
				-aic-num-cores=14
				-onnx-define-symbol=batch_size,1  \
				-multicast-weights  "
		fi
		if [[ "${precision}" == 'fp16' ]]; then
	                compile_options=${compile_options}"  -onnx-define-symbol=batch_size,1 -convert-to-fp16 -aic-num-cores=14"

		fi
        elif [[ "${config}" -eq 2 ]]; then
                if [[ "${precision}" == 'int8' ]]; then
                        compile_options=${compile_options}" -mos=12 \
				-ols=1 \
				-aic-num-cores=12  \
				-quantization-precision=Int8 \
				-quantization-precision-bias=Int32 \
				-quantization-schema=symmetric_with_uint8 \
				-quantization-calibration=Percentile \
				-percentile-calibration-value=99.999 \
				-execute-nodes-in-fp16=Mul,Sqrt,Div,Add,ReduceMean,Split,Softmax,Sub,Gather,Erf,Pow,Concat,Tile,LayerNormalization \
				-onnx-define-symbol=batch_size,1  \
				-multicast-weights  "
                fi
                if [[ "${precision}" == 'fp16' ]]; then
                        compile_options=${compile_options}"   -convert-to-fp16 -mos=14 -ols=1 -aic-num-cores=14  -onnx-define-symbol=batch_size,1    -multicast-weights"

                fi
        fi

	echo ${compile_options}
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
	
	runner_options=' --time 10  -d 0  -i ./inputFiles/input.raw -t '${aic_binary_dir}

	if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'int8' ]]; then
			runner_options=${runner_options}' -a 1 '
                fi
                if [[ "${precision}" == 'fp16' ]]; then
			runner_options=${runner_options}' -a 1 '

                fi
        elif [[ "${config}" -eq 2 ]]; then
                if [[ "${precision}" == 'int8' ]]; then
			runner_options=${runner_options}' -a 1 -S 1 '
                fi
                if [[ "${precision}" == 'fp16' ]]; then
			runner_options=${runner_options}' -a 1 -S 1'
                fi

        fi
	sudo  sudo /opt/qti-aic/exec/qaic-runner ${runner_options}

}


if [[ -f ./weights/yolov7.pt ]]; then
	prepare_model
else
	err "YOLOv7.pt not found. Please make sure yolov7_setup.py is executed" 
	exit 1
fi

generate_inputs
compile_model
run_model
