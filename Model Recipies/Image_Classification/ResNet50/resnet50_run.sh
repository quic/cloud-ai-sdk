###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Use Model Preparator tool for generating ONNX model optimized for AIC100 
#       2. Compile and execute int8 versions of the Model on AIC100 with best recipe for Throughput
#
###########################################################################################################################



CONFIG_FILE=./resnet50_model_info.yaml
aic_binary_dir=./Resnet50_onnx/
INPUT_FILE=./inputFiles/dog.raw


err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
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
prepare_model() {
	source /opt/qti-aic/dev/python/qaic-env/bin/activate
	# WORKSPACE_RESNET50 is the path configured in config file of Model preparator (resnet50_model_info.yaml)
	if [[ -d ./WORKSPACE_RESNET50 ]]; then
		rm -ri ./WORKSPACE_RESNET50
	fi
	sudo python -W ignore /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config "${CONFIG_FILE}"
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

	/opt/qti-aic/exec/qaic-exec \
		-v \
		-aic-hw  \
		-aic-hw-version=2.0 \
		-m=./WORKSPACE_RESNET50/resnet50_v1_preparator_aic100.onnx \
		-aic-binary-dir="${aic_binary_dir}" \
	       	-input-list-file=list.txt \
		-convert-to-quantize \
		-quantization-schema=symmetric_with_uint8 \
		-quantization-precision=Int8 \
		-batchsize=8 \
		-mos=1,2 \
		-ols=4 \
		-aic-num-cores=4 \
		-use-producer-dma=1 \
		-sdp-cluster-sizes=2,2 \
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
run_model() {
	sudo /opt/qti-aic/exec/qaic-runner \
		--time 60 \
		-a 3 \
		--test-data "${aic_binary_dir}" \
		-i "${INPUT_FILE}" \
		-d 1
	}


if [[ -f "${CONFIG_FILE}" ]]; then
	prepare_model
else
	err "Config File for Model Preparator does not exist"
	exit 1
fi

if [[ -f "./WORKSPACE_RESNET50/resnet50_v1_preparator_aic100.onnx" ]]; then
	compile_model
else
        err "ONNX File not generated. Is QAIC Pytools installed?"
        exit 1
fi

run_model
