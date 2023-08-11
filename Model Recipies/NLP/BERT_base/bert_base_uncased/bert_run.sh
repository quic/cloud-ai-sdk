###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Export BERT Large to ONNX model
#       2. Compile and execute Mixed Precision and FP16 versions of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################


#Precision Options: 'mp' or 'fp16'
precision='mp'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=2

aic_binary_dir=./Bert-base-uncased-onnx-128

CONFIG_PATH=bert_base.yaml


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
	if [[ -d ./WORKSPACE ]]; then
		sudo rm -ri ./WORKSPACE
	fi
	sudo python -W ignore /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config $CONFIG_PATH
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
generate_inputs() {
        echo "In generate inputs"
        if [[ -d "./inputFiles" ]]; then
                rm -ri ./inputFiles/*
        fi
        mkdir inputFiles

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'mp' ]]; then
                        python generate_inputs_onnx_model.py --model-path ./WORKSPACE/bert-base-uncased_simplified_preparator_aic100.onnx --batch-size 8

                elif  [[ "${precision}" == 'fp16' ]]; then
                        python generate_inputs_onnx_model.py --model-path ./WORKSPACE/bert-base-uncased_simplified_preparator_aic100.onnx --batch-size 4
                fi


        elif [[ "${config}" -eq 2 ]]; then
                if [[ "${precision}" == 'mp' ]]; then
                        python generate_inputs_onnx_model.py --model-path ./WORKSPACE/bert-base-uncased_simplified_preparator_aic100.onnx --batch-size 1

                elif  [[ "${precision}" == 'fp16' ]]; then
                        python generate_inputs_onnx_model.py --model-path ./WORKSPACE/bert-base-uncased_simplified_preparator_aic100.onnx --batch-size 1
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

        compile_options='-aic-hw -aic-hw-version=2.0 -vvv -compile-only -m=./WORKSPACE/bert-base-uncased_simplified_preparator_aic100.onnx -onnx-define-symbol=sequence,128 -input-list-file=list.txt -aic-binary-dir='${aic_binary_dir}

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'mp' ]]; then
                        compile_options=${compile_options}" -mos=1 \
                                 -ols=4 \
                                 -aic-num-cores=2 \
                                 -quantization-precision=Int8 \
                                 -quantization-precision-bias=Int32 \
                                 -quantization-schema=symmetric_with_uint8 \
                                 -quantization-calibration=Percentile \
                                 -percentile-calibration-value=99.999  \
                                 -execute-nodes-in-fp16=Mul,Sqrt,Div,Add,ReduceMean,Split,Softmax,Sub,Gather,Erf,Pow,Concat,Tile,LayerNormalization \
                                 -stats-batchsize=8 \
				 -onnx-define-symbol=batch_size,8  \
                                 -aic-perf-metrics \
				 -multicast-weights "

                fi
                if [[ "${precision}" == 'fp16' ]]
                then
                        compile_options=${compile_options}"  -convert-to-fp16 -mos=1 -ols=1 -aic-num-cores=2 -onnx-define-symbol=batch_size,4 -stats-batchsize=4   -multicast-weights"

                fi
        elif [[ "${config}" -eq 2 ]]; then
                if [[ "${precision}" == 'mp' ]]; then
                        compile_options=${compile_options}" -mos=7 \
                                 -ols=1 \
                                 -aic-num-cores=7 \
                                 -onnx-define-symbol=batch_size,1 \
                                 -quantization-precision=Int8 \
                                 -quantization-precision-bias=Int32 \
                                 -quantization-schema=symmetric_with_uint8 \
                                 -quantization-calibration=Percentile \
                                 -percentile-calibration-value=99.999  \
                                 -execute-nodes-in-fp16=Mul,Sqrt,Div,Add,ReduceMean,Split,Softmax,Sub,Gather,Erf,Pow,Concat,Tile,LayerNormalization   \
                                 -multicast-weights  \
				 -stats-batchsize=1 \
                                 -aic-binary-dir=${aic_binary_dir} "
                fi
                if [[ "${precision}" == 'fp16' ]]; then
                        compile_options=${compile_options}"   -convert-to-fp16 -mos=14 -ols=1 -aic-num-cores=14  -stats-batchsize=1  -onnx-define-symbol=batch_size,1 -multicast-weights"

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

        runner_options=' --time 10  -d 0  -i ./inputFiles/input_ids.raw -i ./inputFiles/attention_mask.raw -i ./inputFiles/token_type_ids.raw -t '${aic_binary_dir}

        if [[ "${config}" -eq 1 ]]; then
                if [[ "${precision}" == 'mp' ]]; then
                        runner_options=${runner_options}' -a 7 '
                fi
                if [[ "${precision}" == 'fp16' ]]; then
                        runner_options=${runner_options}' -a 7 '

                fi
        elif [[ "${config}" -eq 2 ]]; then
                if [[ "${precision}" == 'mp' ]]; then
                        runner_options=${runner_options}' -a 2 '
                fi
                if [[ "${precision}" == 'fp16' ]]; then
                        runner_options=${runner_options}' -a 7 '
                fi


        fi
        sudo  sudo /opt/qti-aic/exec/qaic-runner ${runner_options}

}


if [[ -f ./bert-base-uncased/generatedModels/bert-base-uncased_simplified.onnx  ]]; then
	prepare_model
else
        err "BERT Base Model not found. Please make sure bert_setup.py is executed"
        exit 1
fi

generate_inputs
compile_model
run_model

