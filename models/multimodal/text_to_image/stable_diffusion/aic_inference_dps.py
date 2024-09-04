"""
Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear
"""
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionInpaintPipeline
import sys
import torch
import numpy as np
from PIL import Image, ImageOps
import time
import argparse
import sys
sys.path.append('/opt/qti-aic/dev/lib/x86_64/')
sys.path.append('./')
import qaicrt

class stable_diffusion():    
    def __init__(self, path, device_id=0):
        self.path = path
        self.device_id = device_id
        self.context = qaicrt.Context()
        pgProperties = qaicrt.QAicProgramGroupProperties()
        # Enable ProgramGroup Properties and set protocol to DataPath
        # For control path -- pgProperties.protocolSelect = qaicrt.QAicProgramGroupPropertiesProtocol.QAIC_PROGRAM_GROUP_PROTOCOL_CONTROL_PATH
        pgProperties.protocolSelect = qaicrt.QAicProgramGroupPropertiesProtocol.QAIC_PROGRAM_GROUP_PROTOCOL_DATA_PATH
        self.program_group = qaicrt.ProgramGroup(self.context, device_id, "Test OverSubscription",
                                    pgProperties)
        self.pipe = OnnxStableDiffusionPipeline.from_pretrained(self.path, provider="CPUExecutionProvider", device_map=self.device_id,program_group=self.program_group,context=self.context)
        self.program_group.enable()
        print('Program Group Enabled')
    
    def get_image(self, prompt, negative_prompt="", seed=3407, n_steps=20):
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        start_time = time.time()
        image = self.pipe(prompt, negative_prompt=negative_prompt, height=512, width=512, num_inference_steps=n_steps).images[0]
        print('total_time: ',time.time()-start_time)
        return image

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Stable Diffusion Text to Image',
                    description = 'Generates image corresponding to the given prompt',
                    epilog = 'Please find the options supported below')
    parser.add_argument('--model_directory', default='stable_diffusion_aic', help='path to the directory containing aic binaries', dest='path')
    parser.add_argument('--prompt', default='Ice Forest with Sun rise', help='text for image generation', dest='prompt')
    parser.add_argument('--device_id', default=0, type=int, help='AIC device id')
    args = parser.parse_args()
    cls_obj = stable_diffusion(args.path,args.device_id)
    #prompt 1
    image = cls_obj.get_image(args.prompt)
    image.save('Prompt1_aic.png')
    cls_obj.program_group.disable(False,0)