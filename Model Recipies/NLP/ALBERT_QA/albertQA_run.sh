#########################################################################################################################################
#!/bin/bash
#
#This script compiles and executes Mixed Precision and FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
########################################################################################################################################

#complete path where model is saved
model_path=./model_dumps/albertQA.onnx

#path to store the AIC binaries
aic_binary_dir=./model_dumps/albertQA-aic

#"throughput"  or "latency"
config="throughput"

#"fp16" or "mixed"     option available only if config is throughput
precision="fp16"

echo "compiling and executing the model for config : ${config}   and  precision : ${precision}"

err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}


#Generating Compile and Execute commands depending on Config
case "$config" in
    "latency") 
	compile_option="-v -aic-hw  \
		-convert-to-fp16 \
		-mos=14 \
		-ols=1 \
		-aic-num-cores=14 \
		-m=${model_path}   \
		-onnx-define-symbol=sequence,128 \
		-onnx-define-symbol=batch,1   \
		-multicast-weights \
		-aic-binary-dir=${aic_binary_dir} \
		-aic-hw-version=2.0 \
		-compile-only"
	execute_option="-a 1 \
                -S 1 \
		-i  inputFiles/input_ids_bs1_sl128.raw \
		-i inputFiles/attention_mask_bs1_sl128.raw \
	        -i inputFiles/token_type_ids_bs1_sl128.raw ";;			      
    
    "throughput")
	if [ $precision == "fp16" ];then
		compile_option="-v -aic-hw  -convert-to-fp16 \
			-mos=4 \
			-ols=1 \
			-aic-num-cores=2\
			-m=${model_path}   \
			-onnx-define-symbol=sequence,128 \
			-onnx-define-symbol=batch,4   \
			-multicast-weights \
			-aic-binary-dir=${aic_binary_dir} \
			-aic-hw-version=2.0  \
			-compile-only" 
		execute_option="-a 7 \
			-i  inputFiles/input_ids_bs4_sl128.raw \
			-i inputFiles/attention_mask_bs4_sl128.raw \
			-i inputFiles/token_type_ids_bs4_sl128.raw "
	fi
	if [[ $precision == "mixed" ]];	then
		compile_option="-v -aic-hw \
			-mos=1 \
			-ols=1 \
			-aic-num-cores=2 \
			-m=${model_path} \
			-onnx-define-symbol=sequence,128 \
			-onnx-define-symbol=batch,8 \
			-input-list-file=input_list_bs8_sl128.txt \
			-profiling-threads=1 \
			-quantization-precision=Int8 \
			-quantization-precision-bias=Int32 \
			-quantization-schema=symmetric_with_uint8 \
			-quantization-calibration=Percentile \
			-percentile-calibration-value=99.999  \
			-execute-nodes-in-fp16=Mul,Sqrt,Div,Add,ReduceMean,Split,Softmax,Sub,Gather,Erf,Pow,Concat,Tile,LayerNormalization  \
			-multicast-weights -stats-batchsize=8 \
			-aic-binary-dir=${aic_binary_dir} \
			-aic-hw-version=2.0  \
			-compile-only"
								
		execute_option="-a 7\
			-i  inputFiles/input_ids_bs8_sl128.raw \
			-i inputFiles/attention_mask_bs8_sl128.raw \
			-i inputFiles/token_type_ids_bs8_sl128.raw "
	fi;;
esac

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
	if [[ -d ${aic_binary_dir} ]];	then
		rm -rf ${aic_binary_dir}
	fi
	/opt/qti-aic/exec/qaic-exec ${compile_option}

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
	sudo /opt/qti-aic/exec/qaic-runner --time 60  \
				   --test-data ${aic_binary_dir} \
				   -d 1 ${execute_option}
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
	if [[ ${config} == "latency" ]]; then
		python generate_inputs_onnx_model.py --model-path ${model_path} --batch-size 1 --sequence 128
	fi
	
	if [[ ${precision} == "mixed" ]] && [[ ${config} == "throughput" ]]; then
		python generate_inputs_onnx_model.py --model-path ${model_path} --batch-size 8 --sequence 128
	fi
	
	if [[ ${precision} == "fp16" ]]  && [[ ${config} == "throughput" ]]; then
		python generate_inputs_onnx_model.py --model-path ${model_path} --batch-size 4 --sequence 128
	fi
	
}
if [[ -f "${model_path}" ]]; then
	generate_inputs
else
	err "Model path not found. Please make sure alberta_setup.sh is executed successfully"
	exit 1
fi

compile_model
execute_model
