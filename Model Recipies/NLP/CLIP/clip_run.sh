###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Use Model Preparator tool for generating ONNX model optimized for AIC100 
#       2. Compile and execute int8 and FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################

#Precision Options: 'int8'or 'fp16'
precision='fp16'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=2

aic_binary_dir=./clip-onnx/

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
	        python generate_inputs_onnx_model.py --model-path  ONNX/clip-vit-base-patch16_split.onnx  --batch-size 4 --sequence 77
	elif [[ "${config}" -eq 2 ]]; then
	        python generate_inputs_onnx_model.py --model-path  ONNX/clip-vit-base-patch16_split.onnx  --batch-size 1 --sequence 77
	fi

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
generate_model(){
        echo "Generating the ONNX model with two outputs[‘logits per image’,‘logits per text’] will take more than an hour!"
	python generate_model.py --save_split_onnx
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
	
	compile_options='-aic-hw -aic-hw-version=2.0 -vvv -compile-only -m=ONNX/clip-vit-base-patch16_split.onnx -input-list-file=list.txt -aic-binary-dir='${aic_binary_dir}
	
	if [[ "${config}" -eq 2 ]]; then
		compile_options=$compile_options" -convert-to-fp16 -mos=6 -ols=1 -aic-num-cores=6 -onnx-define-symbol=Sequence_len,77  -onnx-define-symbol=Num_Class,2  -onnx-define-symbol=Batch_Size,1  -multicast-weights  "
        elif [[ "${config}" -eq 1 ]]; then
		compile_options=$compile_options" -convert-to-fp16 -mos=4 -ols=1 -aic-num-cores=4 -onnx-define-symbol=Sequence_len,77 -onnx-define-symbol=Num_Class,2  -onnx-define-symbol=Batch_Size,4  -multicast-weights "
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
	runner_options=' --time 10  -d 1  -i ./inputFiles/input_ids.raw  -i ./inputFiles/pixel_values.raw -i ./inputFiles/attention_mask.raw  -i ./inputFiles/causal_attention_mask.raw -i ./inputFiles/class_embeds.raw  -t '${aic_binary_dir}
        if [[ "${config}" -eq 2 ]]; then
		runner_options=$runner_options' -a 2 -S 1'
        elif [[ "${config}" -eq 1 ]]; then
		runner_options=$runner_options' -a 3 '
        fi

	sudo /opt/qti-aic/exec/qaic-runner ${runner_options}

	}

generate_model
generate_inputs
compile_model
run_model
