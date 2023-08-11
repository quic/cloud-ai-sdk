###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Generating ONNX model 
#       2. Compile and execute FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################

#Precision Options: 'fp16'
precision='fp16'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=2

aic_binary_dir=./minilm-onnx/

########################################################################
# iGenerate ONNX file 
# Globals:
#   None
# Arguments:
#   None
# Outputs:
#   Generates Optimized ONNX File
########################################################################
prepare_model() {
	export TRANSFORMERS_CACHE=./cache
	python generateModelwithQAHead.py --onnx-output=./generatedModels/ONNX/trained_head/minilm-uncased-squad2.onnx --model-card https://huggingface.co/deepset/minilm-uncased-squad2
	cd ./generatedModels/ONNX/trained_head/
	python -m onnxsim ./minilm-uncased-squad2.onnx ./minilm-uncased-squad2.onnx  --overwrite-input-shape input_ids:1,128 attention_mask:1,128 token_type_ids:1,128 --dynamic-input-shape
	cd ../../../
	sudo rm -rf ./minilm-uncased-squad2-binaries-fp16
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

	if [[ "${config}" -eq 1 ]]; then
		python generate_inputs_onnx_model.py --model-path ./generatedModels/ONNX/trained_head/minilm-uncased-squad2.onnx --batch-size 8
        elif [[ "${config}" -eq 2 ]];	then
		python generate_inputs_onnx_model.py --model-path ./generatedModels/ONNX/trained_head/minilm-uncased-squad2.onnx --batch-size 1
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
compile_model(){
	if [[ -d ${aic_binary_dir} ]]
	then
		rm -ri ${aic_binary_dir}
	fi

	        compile_options='-aic-hw -aic-hw-version=2.0 -vvv -compile-only -m=./generatedModels/ONNX/trained_head/minilm-uncased-squad2.onnx -aic-binary-dir='${aic_binary_dir}

        if [ $config == 2 ]
        then
                if [ $precision == 'fp16' ]
                then
                        compile_options=$compile_options"  -convert-to-fp16 -mos=4 -ols=1 -aic-num-cores=4  -onnx-define-symbol=sequence,128  -onnx-define-symbol=batch,1 -multicast-weights -input-list-file=list.txt "
		fi
        elif [ $config == 1 ]
	then
		if [ $precision == 'fp16' ]
                then
                        compile_options=$compile_options"  -convert-to-fp16 -mos=1 -ols=4 -aic-num-cores=2   -onnx-define-symbol=sequence,128 -input-list-file=list.txt  -onnx-define-symbol=batch,8  -multicast-weights "
		fi

	fi

        /opt/qti-aic/exec/qaic-exec $compile_options

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
run_model(){

	 runner_options=' --time 10  -d 1 -i ./inputFiles/input_ids.raw -i ./inputFiles/attention_mask.raw -i ./inputFiles/token_type_ids.raw  -t '${aic_binary_dir}

        if [ $config == 1 ]
        then
                if [ $precision == 'fp16' ]
                then
                        runner_options=$runner_options' -a 7'
                fi
        elif [ $config == 2 ]
        then
                if  [ $precision == 'fp16' ]
                then
			runner_options=$runner_options' -a 3'
		fi
	fi
	sudo /opt/qti-aic/exec/qaic-runner $runner_options

}


prepare_model
generate_inputs
compile_model
run_model

