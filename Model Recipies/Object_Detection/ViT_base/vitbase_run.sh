###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Export ViT-Base to ONNX model
#       2. Compile and execute FP16 version of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################

#Precision Options:  'fp16'
precision='fp16'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=2


aic_binary_dir=./vitbase-onnx

########################################################################
# Generate ONNX file optimized for AIC100.
# Globals:
#   None
# Arguments:
#   None
# Outputs:
#   Generates Optimized ONNX File
########################################################################
prepare_model() {
	cd generatedModels/
	sudo sh generateModels.sh

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

	 if [[ "${precision}" == 'fp16' ]]; then
		  if [[ "${config}" -eq 1 ]];  then
			  sudo python generateSampleInput.py --img_path inputFiles/000000039769.jpg --batch_size 12 --output_path ./inputFiles/000000039769_1_12_224_224.raw
		  elif [[ "${config}" -eq 2 ]]; then
			  sudo python generateSampleInput.py --img_path inputFiles/000000039769.jpg --batch_size 1 --output_path ./inputFiles/000000039769_1_12_224_224.raw
		  fi
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
 

                compile_options='-aic-hw -aic-hw-version=2.0 -vvv -compile-only -m=./generatedModels/vit_base_16_224.onnx  -aic-binary-dir='${aic_binary_dir}

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'fp16' ]]; then
                        compile_options=$compile_options"   -convert-to-fp16 -mos=1 -ols=1 -aic-num-cores=2 -multicast-weights -onnx-define-symbol=batch_size,12  "
                fi


        elif [[ "${config}" -eq 2 ]]; then
                if  [[ "${precision}" == 'fp16' ]];  then
                        compile_options=$compile_options"  -convert-to-fp16 -mos=12 -ols=1 -aic-num-cores=12 -multicast-weights -onnx-define-symbol=batch_size,1 "
                fi
        fi
        /opt/qti-aic/exec/qaic-exec ${compile_options}

	}

run_model() {
	        runner_options=' --time 10  -d 1 -i ./inputFiles/000000039769_1_12_224_224.raw  -t '${aic_binary_dir}

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'fp16' ]];  then
                        runner_options=$runner_options' -a 7'
                fi

        elif [[ "${config}" -eq 2 ]]; then
                if  [[ "${precision}" == 'fp16' ]]; then
                        runner_options=${runner_options}' -a 1 -S 1'
                fi
        fi


        sudo /opt/qti-aic/exec/qaic-runner ${runner_options}

	}


pip install -r requirements.txt
prepare_model
generate_inputs
compile_model
run_model
