###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Export R-CNN Model to ONNX model
#       2. Compile and execute Mixed Precision and FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################

#Precision Options: 'mp' or 'fp16'
precision='fp16'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=2


aic_binary_dir=./rcnn-onnx/

########################################################################
# Generate ONNX file optimized for AIC100.
# Globals:
#   Config File for Model Preparator
# Arguments:
#   None
# Outputs:
#   Generates Optimized ONNX File
########################################################################
prepare_model() {
        cd ./scripts
        python edit_onnx_file.py -model ../detectron2/tools/deploy/onnx_model_maskrcnn/model.onnx
        cd ../
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
	if [[ -d ${aic_binary_dir} ]]; then
		rm -ri ${aic_binary_dir}
	fi

	compile_options='-aic-hw -aic-hw-version=2.0 -vvv -compile-only -m=./scripts/model_modified.onnx -input-list-file=input-list.txt -aic-binary-dir='${aic_binary_dir}

	if [[ "${config}" -eq 1 ]]; then
		if [[ "${precision}" == 'mp' ]]; then
			compile_options=$compile_options" -mos=1 \
				-ols=1 \
				-aic-num-cores=2 \
				-multicast-weights \
				-execute-nodes-in-fp16=GenerateProposals,CollectRpnProposals,DistributeFpnProposals,BboxTransform,RoiAlign,BoxWithNMSLimit \
				-quantization-schema=symmetric_with_uint8 \
				-quantization-precision=Int8  \
				-aic-pmu-recipe=KernelUtil "
		elif [[ "${precision}" == 'fp16' ]]; then
			 compile_options=$compile_options"  -convert-to-fp16 -mos=1 -ols=1 -aic-num-cores=2  -multicast-weights "
		fi

        elif [[ "${config}" -eq 2 ]]; then
	        if [[ "${precision}" == 'mp' ]];  then
                        compile_options=$compile_options"  -mos=1 \
				-ols=1 \
				-aic-num-cores=12 \
				-multicast-weights \
				-execute-nodes-in-fp16=GenerateProposals,CollectRpnProposals,DistributeFpnProposals,BboxTransform,RoiAlign,BoxWithNMSLimit \
				-quantization-schema=symmetric_with_uint8 \
				-quantization-precision=Int8  \
				-aic-pmu-recipe=KernelUtil  "
                elif [[ "${precision}" == 'fp16' ]];  then
                         compile_options=$compile_options"  -convert-to-fp16 -mos=1 -ols=4 -aic-num-cores=14 -multicast-weights "
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
	runner_options=' --time 10  -d 1   -i ./000000000139_in_data.raw -i ./000000000139_in_im_info.raw  -t '${aic_binary_dir}
        if [[ "${config}" -eq 2 ]]; then
		runner_options=${runner_options}' -a 1 -S 1'
        elif [[ "${config}" -eq 1 ]]; then
		runner_options=${runner_options}' -a 7 '
        fi

	sudo  sudo /opt/qti-aic/exec/qaic-runner ${runner_options}

	}

prepare_model
compile_model
run_model
