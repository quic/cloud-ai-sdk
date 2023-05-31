###############################################################################
# Run model preparator tool.
###############################################################################

#!/bin/bash
#complete modelpath
model_output_path=./generatedModels/ONNX/cased/prepmodel.onnx

#preparing the model as per AIC
prepare_model(){
	if [ -d WORKSPACE ]
	then
		rm -rf WORKSPACE
	fi

	python3 /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config bertbase.yaml
	python3 generateModel.py --model-ip-path  WORKSPACE/bert-base-cased_fix_outofrange_fp16_preparator_aic100.onnx --model-op-path ${model_output_path}

}

prepare_model
