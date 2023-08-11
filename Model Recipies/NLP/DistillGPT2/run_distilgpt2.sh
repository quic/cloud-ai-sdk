###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Use generateModel.py and splitAndSimplifyONNXGraph.py  for generating ONNX model optimized for AIC100 
#       2. Compile and execute FP16 version of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################

#path to onnx model
model_dir=./generatedModels/distilgpt2_onetoken_sim.onnx


#path to save the aic binaries
aic_binary_dir=./generatedModels/distilgpt2_aic

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
	python generateInputs.py --model-path $model_dir --batch-size 64  --sequence 8
}

########################################################################
# Use  Use generateModel.py and splitAndSimplifyONNXGraph.py to generate ONNX file optimized for AIC100.
# Globals:
#   None
# Arguments:
#   None
# Outputs:
#   Generates Optimized ONNX File and inputs for the model
########################################################################
prepare_model(){
	pip install -r requirements.txt
	mkdir -p ./generatedModels/
	git clone https://github.com/huggingface/transformers.git
	cd transformers/
        git checkout 85d69a7dd1c29f9b9bca7b5a9e6b1319caf07c6b
	git apply ../slice_before_logits.patch
  	pip install -e .
	cd ../
	python generateModel.py
	python splitAndSimplifyONNXGraph.py --onnx_input ./generatedModels/Onnx/distilgpt2_onetoken.onnx --onnx_output   $model_dir
	generate_inputs
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
	if [ -d ${aic_binary_dir} ]
	then
		rm -rf ${aic_binary_dir}
	fi

	#compliling the model
	/opt/qti-aic/exec/qaic-exec -v -aic-hw  \
				-convert-to-fp16 \
				-mos=1 \
				-ols=1 \
				-aic-num-cores=2 \
				-m=$model_dir   \
				-onnx-define-symbol=seq_len,8 \
				-onnx-define-symbol=batch_size,64  \
				-multicast-weights  \
				-stats-batchsize=64 \
				-aic-binary-dir=$aic_binary_dir \
				-aic-hw-version=2.0 \
				-compile-only

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
execute_model(){
	
	#executing the model
	sudo /opt/qti-aic/exec/qaic-runner --time 60 -a 7 \
				 --test-data $aic_binary_dir  \
				 -i inputFiles/input_ids_bs64_sl8.raw \
				 -i inputFiles/position_ids_bs64_sl8.raw \
				 -i inputFiles/attention_mask_bs64_sl8.raw \
				 -d 2

}



prepare_model

if [[ -f $model_dir ]]; then	
	compile_model
else
	err "onnx model is not generated "
	exit 1
fi


execute_model
