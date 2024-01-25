# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os.path

from diffusers import OnnxDeciDiffusionPipeline
import time
import argparse
import sys

sys.path.append('/opt/qti-aic/dev/lib/x86_64/')
sys.path.append('./')
import qaicrt


class Diffusion:
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

        self.pipe = OnnxDeciDiffusionPipeline.from_pretrained(self.path, provider="CPUExecutionProvider", device_map=self.device_id,
                                                              program_group=self.program_group, context=self.context)

        self.program_group.enable()
        print('Program Group Enabled')

    def get_image(self, *, prompt):
        start_time = time.time()
        img = self.pipe(prompt).images[0]
        return img, time.time() - start_time


def main(args):
    print(f"Generating {args.prompt}")
    pipe = Diffusion(args.aic_dir, args.device_id)
    if args.repetitions == 1:
        print(f"Note: to benchmark, please execute the generation cycle args.repetitions>5 times, and average the runtime of the last k-1 cycles.")
    latencies = list()
    for sample in range(args.repetitions):
        image, duration = pipe.get_image(prompt=args.prompt)
        latencies.append(duration)
        image.save(os.path.join(args.output_path, args.prompt.replace(' ', '') + f'{sample}.png'))
    if args.repetitions > 1:
        print(f"Benchmark: {sum(latencies[1:]) / (args.repetitions - 1):.2f} seconds/image")
    
    pipe.program_group.disable(False, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Diffusion Text to Image',
        description='Generates image corresponding to the given prompt',
        epilog='Please find the options supported below')
    parser.add_argument('--aic_dir', required=True, help='path to the directory containing aic binaries', dest='aic_dir')
    parser.add_argument('--prompt', default='', help='text for image generation', dest='prompt')
    parser.add_argument('--output_path', required=True, help='path to store generated images', dest='output_path')
    parser.add_argument('--repetitions', required=False, default=1, type=int, help='benchmark repetitions', dest='repetitions')
    parser.add_argument('--device_id', default=0, type=int, help='AIC device id')
    main(parser.parse_args())
