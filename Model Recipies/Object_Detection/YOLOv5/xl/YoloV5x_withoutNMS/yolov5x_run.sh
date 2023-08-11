###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Export YOLOv5-xl to ONNX model without NMS
#       2. Compile and execute int8 and FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################

#Precision Options: 'int8'or 'fp16'
precision='int8'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=2


aic_binary_dir=./QPC_DIR

########################################################################
# Install dependencies for YOLOv5l
# Arguments:
#   None
# Outputs:
#   None
########################################################################
yolov5_setup(){
	pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
	pip install ninja
}

########################################################################
# Generate ONNX file for YOLOv5
# Arguments:
#   None
# Outputs:
#   Generates ONNX File without NMS
########################################################################
prepare_model() {
	cd generatedModels
	sudo sh ./generateModels.sh 5x 416 416
	cd ..
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
        python generate_inputs_onnx_model.py --model-path ./generatedModels/ONNX/yolov5x_416_416_without_abp_nms.onnx --batch-size 1
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
	if [[ -d ${aic_binary_dir} ]];	then
		rm -ri ${aic_binary_dir}
	fi

	        compile_options='-aic-hw -aic-hw-version=2.0 -vvv -compile-only -m=./generatedModels/ONNX/yolov5x_416_416_without_abp_nms.onnx  -aic-binary-dir='${aic_binary_dir}

        if [[ ${config} -eq 1 ]]; then
                if [[ ${precision} == 'fp16' ]]; then
                        compile_options=$compile_options"  -convert-to-fp16 -mos=1 -ols=2 -aic-num-cores=1 -input-list-file=./inputdatafile.txt -onnx-define-symbol=batch_size,1 "
                fi

                if [[ ${precision} == 'int8' ]];  then
                        compile_options=$compile_options" \
				-convert-to-quantize \
				-quantization-schema=symmetric_with_uint8 \
				-quantization-precision=Int8 \
				-mos=1 \
				-ols=2 \
				-aic-num-cores=1 \
				-input-list-file=./inputdatafile.txt \
				-onnx-define-symbol=batch_size,1"
                fi

        elif [[ ${config} -eq 2 ]]; then
                if  [[ ${precision} == 'fp16' ]]; then
                        compile_options=$compile_options"  -convert-to-fp16 -mos=1 -ols=1 -aic-num-cores=12 -input-list-file=./inputdatafile.txt -multicast-weights -onnx-define-symbol=batch_size,1 "
                fi
                if [[ ${precision} == 'int8' ]]; then
                         compile_options=$compile_options"  \
				 -convert-to-quantize \
				 -quantization-schema=symmetric_with_uint8 \
				 -quantization-precision=Int8 \
				 -mos=2 \
				 -ols=1 \
				 -aic-num-cores=6 \
				 -input-list-file=./inputdatafile.txt \
				 -multicast-weights \
				 -onnx-define-symbol=batch_size,1 "
                fi
	fi 
	/opt/qti-aic/exec/qaic-exec ${compile_options}



}

run_model() {

	runner_options=' --time 10  -d 1 -i ./inputFiles/images.raw  -t '${aic_binary_dir}

        if [[ ${config} -eq 1 ]]; then
                if [[ ${precision} == 'fp16' ]]; then
                        runner_options=$runner_options' -a 14'
                fi

                if [[ ${precision} == 'int8' ]]; then
                        runner_options=$runner_options' -a 14'
                fi
        elif [[ ${config} -eq 2 ]]; then
                if  [[ ${precision} == 'fp16' ]];  then
                        runner_options=$runner_options' -a 1 -S 1'
                fi
                if [[ ${precision} == 'int8' ]]; then
                        runner_options=$runner_options' -a 2 -S 1'
                fi
        fi


        sudo /opt/qti-aic/exec/qaic-runner $runner_options

}

yolov5_setup
prepare_model
generate_inputs
compile_model
run_model
