###############################################################################
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
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
	python3 generateModel.py --model-ip-path  WORKSPACE/model_preparator_aic100.onnx --model-op-path ${model_output_path}

}

prepare_model
