###########################################################################################################################
#!/bin/bash
#
#This script does following:
#       1. Generate ONNX model 
#       2. Compile and execute FP16 version of the Model on AIC100 with best recipes for Throughput and Latency
#
###########################################################################################################################

#Precision Options: 'fp16'
precision='fp16'

#Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=2


aic_binary_dir=./gpt2-onnx
input_files_dir=inputFiles



########################################################################
# Install requirements and Export model to ONNX.
# Globals:
#   Config File for Model Preparator
# Arguments:
#   None
# Outputs:
#   Generates  ONNX File
########################################################################
prepare_model() {
    pip install -r requirements.txt
    python generateModel.py --model-name gpt2
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

        compile_options='-aic-hw -aic-hw-version=2.0 -vvv -compile-only -onnx-define-symbol=seq_len,8 -onnx-define-symbol=total_seq_len,9  -onnx-define-symbol=past_seq_len,1 -m=./gpt2.onnx -aic-binary-dir='${aic_binary_dir}

        if [[ "${config}" -eq 1 ]];  then
                if [[ "${precision}" == 'fp16' ]];  then
                        compile_options=$compile_options"  -convert-to-fp16 -mos=7 -ols=1 -aic-num-cores=7 -input-list-file=list.txt -onnx-define-symbol=batch_size,14"
                fi
	elif [[ "${config}" -eq 2 ]]; then
                if  [[ "${precision}" == 'fp16' ]];  then
                        compile_options=$compile_options"  -convert-to-fp16 -mos=4 -ols=1 -aic-num-cores=14 -input-list-file=list_bs1.txt  -onnx-define-symbol=batch_size,1"
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
        runner_options=' --time 10  -d 1  -t '${aic_binary_dir}

        if [ $config == 1 ]
        then
                if [ $precision == 'fp16' ]
                then
                        runner_options=$runner_options' -a 2 \
				-i ./inputFiles/input_ids.raw \
				-i ./inputFiles/position_ids.raw \
				-i ./inputFiles/attention_mask.raw \
				-i ./inputFiles/past_0.raw \
				-i ./inputFiles/past_1.raw \
				-i ./inputFiles/past_2.raw \
				-i ./inputFiles/past_3.raw \
				-i ./inputFiles/past_4.raw \
				-i ./inputFiles/past_5.raw \
				-i ./inputFiles/past_6.raw \
				-i ./inputFiles/past_7.raw \
				-i ./inputFiles/past_8.raw \
				-i ./inputFiles/past_9.raw \
				-i ./inputFiles/past_10.raw \
				-i ./inputFiles/past_11.raw'
                fi
        elif [[ "${config}" -eq 2 ]];  then
                if  [[ "${precision}" == 'fp16' ]];  then
                        runner_options=$runner_options' -i ./inputFiles_bs1/input_ids.raw \
				-i ./inputFiles_bs1/position_ids.raw \
				-i ./inputFiles_bs1/attention_mask.raw \
				-i ./inputFiles_bs1/past_0.raw \
				-i ./inputFiles_bs1/past_1.raw \
				-i ./inputFiles_bs1/past_2.raw \
				-i ./inputFiles_bs1/past_3.raw \
				-i ./inputFiles_bs1/past_4.raw \
				-i ./inputFiles_bs1/past_5.raw \
				-i ./inputFiles_bs1/past_6.raw \
				-i ./inputFiles_bs1/past_7.raw \
				-i ./inputFiles_bs1/past_8.raw \
				-i ./inputFiles_bs1/past_9.raw \
				-i ./inputFiles_bs1/past_10.raw \
				-i ./inputFiles_bs1/past_11.raw \
				-a 1 \
				-S 1'
                fi
	fi
        sudo /opt/qti-aic/exec/qaic-runner $runner_options
}

prepare_model
compile_model
run_model

