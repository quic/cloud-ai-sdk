###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Export Roberta-base model to ONNX model
#       2. Compile and execute Mixed Precision and FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################

#requirments
sudo pip install -r requirements.txt



#onnx model path
model_path=./generatedModels/roberta_base.onnx

#path to store the aic binary files
aic_binary_dir=generatedModels/roberta_aic_fp16/


# "latency" or "throughput"
config="throughput"

err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}

#Create Compile and execute commands based on configuration

case $config in 
	"latency")
		compile_option="-v -aic-hw  \
			-convert-to-fp16 \
			-mos=14 \
			-ols=1 \
			-aic-num-cores=14 \
			-m=${model_path}   \
			-onnx-define-symbol=sequence,128 \
			-onnx-define-symbol=batch,1 \
			-multicast-weights  \
			-aic-binary-dir=${aic_binary_dir} \
			-aic-hw-version=2.0  \
			-compile-only"
		execute_option=" -a 1  \
			-i inputFiles/input_id.raw \     
			-i inputFiles/attention_mask.raw      \
			-S 1"
		;;
	"throughput")
		compile_option="-v -aic-hw  \
			-convert-to-fp16 \
			-mos=1 \
			-ols=4 \
			-aic-num-cores=4 \
			-m=${model_path}   \
			-onnx-define-symbol=sequence,128 \
			-onnx-define-symbol=batch,8 \
			-multicast-weights  \
			-aic-binary-dir=${aic_binary_dir} \
			-aic-hw-version=2.0 \
			-compile-only"
		execute_option="-a 3  \
			-i inputFiles/input_id.raw \
			-i inputFiles/attention_mask.raw"
		;;

esac


generate_inputs(){
	if [[ ${config} == "latency" ]]; then
		python generateSampleInput.py --sl 128 --bs 1
	elif [[ ${config} == "throughput" ]]; then
		python generateSampleInput.py --sl 128 --bs 8
	fi
}


prepare_model(){
	mkdir ./generatedModels/
	echo ${model_path}

	sudo python Model.py --model-card deepset/roberta-base-squad2 --output-path ${model_path}
	generate_inputs
}

compile_model(){
	if [[ -d ${aic_binary_dir} ]];	then
		rm -rf ${aic_binary_dir}
	fi
	/opt/qti-aic/exec/qaic-exec $compile_option
}

execute_model(){
	sudo /opt/qti-aic/exec/qaic-runner --time 60  --test-data ${aic_binary_dir}  -d 1 $execute_option

}

prepare_model
compile_model
execute_model
