'''
Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import qaic
import numpy as np
import argparse

# Establish arguments to accept
def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model-path",
        dest='model_path',
        default=
        '/opt/qti-aic/integrations/qaic_onnxrt/tests/resnet50/resnet50-v1-12-batch.onnx',
        help='Pass path to qpc of this model to avoid compilation')

    parser.add_argument(
        "--config-path",
        dest='config_path',
        default=
        './resnet_config.yaml',
        help='Pass path to qpc of this model to avoid compilation')
    
    parser.add_argument(
        "--input",
        dest='input_img',
        help=
        'If image is not provided, random values will be generated as input. Input image should be 1*3*224*224 pixel in raw format'
    )

    parser.add_argument(
        "--num_iters",
        dest='num_iters',
        default=1000,
        help='Enter number of inferences you want to run on the model')
    
    return parser.parse_args()

def main(args):
    
    resnet_sess = qaic.Session(
        args.model_path,
        options_path=args.config_path,
        enable_profiling=True)

    input_shape, input_type = resnet_sess.model_input_shape_dict['data']

    # Read input

    if args.input_img is None:
        x = np.random.randn(*input_shape).astype(input_type)
    else:
        img = np.fromfile(args.input_img, dtype=input_type)
        x = np.resize(img, input_shape)

    # Run inference
    input_dict = {'data': x}

    for _ in range(args.num_iters):
        resnet_sess.run(input_dict)
    
    print('\n\n\n\n-------------- Metrics --------------\n\n\n\n')
    resnet_sess.print_metrics()
    print('\n\n\n\n-------------- Profile Data --------------\n\n\n\n')
    resnet_sess.print_profile_data(n=5)
    metrics = resnet_sess.get_metrics()

if __name__ == '__main__':
    args = get_args()
    main(args)