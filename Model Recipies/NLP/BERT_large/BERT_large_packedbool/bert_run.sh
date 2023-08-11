###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Export BERT Large to ONNX model
#       2. Compile and execute Mixed Precision and FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################


#Precision Options: 'mp'or 'fp16'
precision='mp'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

aic_binary_dir=./Bert-Large-packed-onnx-384

err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}

########################################################################
# Generate ONNX file for BERT Large
# Arguments:
#   None
# Outputs:
#   Generates Optimized ONNX File
########################################################################

prepare_model() {
	#Generate onnx file for BERT Large
	mkdir -p ./generatedModels/ONNX/
	sudo python3 -W ignore Model.py --config ./mlCommonsBertFiles/bert_config.json --checkpoint ./mlCommonsBertFiles/model.ckpt-5474 --save-onnx-variable-sl

	#Generate Bert model for packed inputs
	sudo python3 -W ignore modifyBertForPacked.py

	#Generate Packed inputs
	sudo python3 generatePackedInputs.py
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

	compile_options='-aic-hw -aic-hw-version=2.0 -vvv -compile-only -m=./generatedModels/ONNX/BERT_MLCommons_Flexible_BS_SL_Packed.onnx -onnx-define-symbol=seq_length,384 -onnx-define-symbol=seg_length,384 -input-list-file=list.txt -aic-binary-dir='${aic_binary_dir}

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'mp' ]]; then
			compile_options=$compile_options"  -mos=1 \
				-ols=2 \
				-aic-num-cores=1 \
				-size-split-granularity=1536 \
				-vtcm-working-set-limit-ratio=1 \
				-quantization-precision=Int8 \
				-quantization-precision-bias=Int32 \
				-quantization-schema=symmetric_with_uint8 \
				-quantization-calibration=Percentile \
				-percentile-calibration-value=99.9977 \
				-execute-nodes-in-fp16=Add,Div,Erf,Softmax \
				-onnx-define-symbol=batch_size,1 \
				-multicast-weights "
                fi
                if [[ "${precision}" == 'fp16' ]]
                then
	                compile_options=$compile_options" -convert-to-fp16 -mos=4 -ols=1 -aic-num-cores=4 -onnx-define-symbol=batch_size,1   -multicast-weights"

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

	runner_options=' --time 10  -d 1  -i ./inputFiles/packed/input_ids.raw -i ./inputFiles/packed/input_mask.raw  -i ./inputFiles/packed/segment_ids.raw -i ./inputFiles/packed/input_position_ids.raw -t '${aic_binary_dir}

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'mp' ]]; then
                        runner_options=${runner_options}' -a 14 '
                fi
                if [[ "${precision}" == 'fp16' ]]; then
                        runner_options=${runner_options}' -a 3 '

                fi
	fi
	sudo /opt/qti-aic/exec/qaic-runner $runner_options
}



if [[ -f ./mlCommonsBertFiles/bert_config.json && -f ./mlCommonsBertFiles/model.ckpt-5474.data-00000-of-00001 && -f ./mlCommonsBertFiles/model.ckpt-5474.index && -f ./mlCommonsBertFiles/model.ckpt-5474.meta && -f ./mlCommonsBertFiles/model.pytorch && -f ./mlCommonsBertFiles/vocab.txt ]]; then
        prepare_model
else
        err "BERT Large Model not found. Please make sure bert_setup.py is executed"
        exit 1
fi

compile_model
run_model

