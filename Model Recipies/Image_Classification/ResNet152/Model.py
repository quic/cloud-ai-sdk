##############################################################################
#
#Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#All Rights Reserved.
#Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#All data and information contained in or disclosed by this document are
#confidential and proprietary information of Qualcomm Technologies, Inc., and
#all rights therein are expressly reserved. By accepting this material, the
#recipient agrees that this material and the information contained therein
#are held in confidence and in trust and will not be used, copied, reproduced
#in whole or in part, nor its contents revealed in any manner to others
#without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################

import os
import torch
import torchvision.models as models

model = models.resnet152(pretrained=True)
model.eval()

os.makedirs("./ONNX/", exist_ok=True)

dummy_input = torch.randn(1, 3, 224, 224).type(torch.FloatTensor).to('cpu')
torch.onnx.export(model, dummy_input, "./ONNX/ResNet152.onnx")

