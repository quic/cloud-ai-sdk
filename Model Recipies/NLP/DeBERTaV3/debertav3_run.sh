###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Use generateModel.py  for generating ONNX model optimized for AIC100 
#       2. Compile and execute in  FP16 version of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################


#path to the onnx model
model_dir=./generatedModels/Onnx/microsoft/deberta-v3-xsmall-classification.onnx

#path to store the AIC bin files
aic_binary_dir=./generatedModels/debertav3_xsmall_aic/


#"latency" or "throughput"
config="throughput"

err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}


case $config in
	"latency")
		compile_option="-v -aic-hw  \
			-convert-to-fp16 \
			-mos=7 \
			-ols=1 \
			-aic-num-cores=7 \
			-m=${model_dir}   \
			-onnx-define-symbol=sequence,128 \
			-onnx-define-symbol=batch,1 \
			-stats-batchsize=1 \
			-multicast-weights \
			-aic-binary-dir=${aic_binary_dir} \
			-aic-hw-version=2.0  \
			-compile-only"
		
		execute_option="-a 2 \
			-i inputFiles/input_ids_bs1.raw \
			-i inputFiles/attention_mask_bs1.raw -S 1"
		;;
	
	"throughput")
		compile_option="-v -aic-hw  \
			-convert-to-fp16 \
			-mos=1 \
			-ols=2 \
			-aic-num-cores=2 \
			-m=${model_dir}   \
			-onnx-define-symbol=sequence,128 \
			-onnx-define-symbol=batch,4 \
			-stats-batchsize=4 \
			-multicast-weights \
			-aic-binary-dir=${aic_binary_dir} \
			-aic-hw-version=2.0  \
			-compile-only"
		execute_option="-a 7   \
			-i inputFiles/input_ids_bs4.raw \
			-i inputFiles/attention_mask_bs4.raw"
		;;
	


esac
	
  
  
########################################################################
# Uses generateModel.py  to generate ONNX file optimized for AIC100.
# Globals:
#   None
# Arguments:
#   bs : batchsize
# Outputs:
#   Generates Optimized ONNX File and random input file
########################################################################

prepare_model(){
	pip install -r requirements.txt
	if [ $config == "latency" ]
	then
		python generateModel.py --save_onnx --save_raw_files --bs 1 
	fi
	
	if [ $config == "throughput" ]
	then
		python generateModel.py --save_onnx --save_raw_files --bs 4 
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
	if [ -d ${aic_binary_dir} ]
	then
		rm -rf ${aic_binary_dir}
	fi
	/opt/qti-aic/exec/qaic-exec $compile_option
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
	sudo /opt/qti-aic/exec/qaic-runner --time 60 \
				 --test-data ${aic_binary_dir}  \
				 -d 3   $execute_option	
}									 




if [[ -f "generateModel.py" ]]; then
	prepare_model
else
	err "generateModel.py  does not exist"
	exit 1
fi

if [[ -f ${model_dir} ]]; then
	compile_model
else
        err "ONNX File not generated. check model has been genereated or not"
        exit 1
fi


execute_model

