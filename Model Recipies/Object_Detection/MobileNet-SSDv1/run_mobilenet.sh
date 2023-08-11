###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Export SSDMobileNetv1 to ONNX model using Model Preparator tool
#       2. Compile and execute INT8 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################

#path of the onnx model
model_dir=generatedModels/SSDMobileNetV1_300_300_Without_ABP_NMS_batcsize_sim.onnx

#path to save the aic bin files
aic_binary_dir=generatedModels/ssdmobilenetv1_aic_int8


########################################################################
# Use Model Preparator tool to generate ONNX file optimized for AIC100.
# Globals:
#   Config File for Model Preparator
# Arguments:
#   None
# Outputs:
#   Generates Optimized ONNX File
########################################################################
prepare_model()
{
	if [[ -d WORKSPACE ]]; then
		rm -rf WORKSPACE
	fi
	
	mkdir -p generatedModels
	source /opt/qti-aic/dev/python/qaic-env/bin/activate
	python /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config ssd_mv1_model_info_smart_nms.yaml
	python -W ignore splitAndSimplifyONNXGraph.py --onnx_input WORKSPACE/ssd_mobilenet_v1_10_preparator_aic100.onnx  --onnx_output $model_dir
	python generateInputs.py
	deactivate
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
quantize_compile_model(){
	if [[ -d ${aic_binary_dir} ]];	then
		rm -rf ${aic_binary_dir}
	fi
	
	/opt/qti-aic/exec/qaic-exec -v -aic-hw  \
				 -convert-to-quantize \
				 -quantization-schema=symmetric_with_uint8 \
				 -quantization-precision=Int8 \
				 -batchsize=4 \
				 -mos=1 \
				 -ols=8 \
				 -aic-num-cores=1 \
				 -m=${model_dir}    \
				 -onnx-define-symbol=batch_size,4 \
				 -input-list-file=input_list.txt  \
				 -size-split-granularity=1048 \
				 -vtcm-working-set-limit-ratio=0.75 \
				 -enable-channelwise \
				 -quantization-schema=symmetric_with_uint8 \
				 -quantization-precision=Int8 \
				 -allocator-dealloc-delay=2  \
				 -aic-binary-dir=${aic_binary_dir} \
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
	sudo /opt/qti-aic/exec/qaic-api-test --time 60 -a 14 \
				 --test-data=${aic_binary_dir} \
				 -T 8  \
				 -i inputFiles/input_raw_0.raw \
				 -d 2

}



prepare_model
quantize_compile_model
execute_model
