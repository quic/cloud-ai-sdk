import torch
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.getcwd(), "yolov7"))

from models.experimental import attempt_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        type=str,
                        default='yolov7',
                        help='weight path path')
    parser.add_argument("--h_w",
                        type=int,
                        nargs='+',
                        default=[640, 640],
                        help="Give Input resolutions, height, width")
    args = parser.parse_args()

    device = 'cpu'
    print(f"Generating onnx for {args.name} model")
    model = attempt_load(f"../weights/{args.name}.pt", map_location=device)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, args.h_w[0], args.h_w[1]),
        f'ONNX/{args.name}_{args.h_w[0]}_{args.h_w[1]}_smartNMS.onnx',
        opset_version=11,
        input_names=['images'])
