##############################################################################
#
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################
import argparse
import onnxruntime as ort
import torch
import os
import cv2
import numpy as np
import shutil
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "../yolov7"))
from utils.general import *
from utils.datasets import *
from utils.plots import *

yolov7_anchors = [
    [[12, 16], [19, 36], [40, 28]],  #p/8
    [[36, 75], [76, 55], [72, 146]],  #p/16
    [[142, 110], [192, 243], [459, 401]]  #p/32
]

yolov7_e6e_anchors = [
    [[19, 27], [44, 40], [38, 94]],  #p2/8
    [[96, 68], [86, 152], [180, 137]],  #p/16
    [[140, 301], [303, 264], [238, 542]],  #p/32
    [[436, 615], [739, 380], [925, 792]]  #p/64
]


class YOLOLayer(torch.nn.Module):

    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride,
                 fmap):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = fmap[0], fmap[
            1], 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.create_grids((self.nx, self.ny))

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)
        yv, xv = torch.meshgrid([
            torch.arange(self.ny, device=device),
            torch.arange(self.nx, device=device)
        ])
        self.grid = torch.stack((xv, yv), 2).view(
            (1, 1, self.ny, self.nx, 2)).float()
        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):

        io = p.sigmoid()
        io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
        io[..., 2:4] = (io[..., 2:4] * 2)**2 * self.anchor_wh
        io[..., :4] *= self.stride

        return io.view(1, -1, self.no)


def inference(img, opt):

    imgsz = (opt.img_size, opt.img_size)

    if opt.model_name == "yolov7":
        anchors = yolov7_anchors
        feature_map_size = [80, 40, 20]
        downsize_factor = [8, 16, 32]
    elif opt.model_name == "yolov7-e6e":
        anchors = yolov7_e6e_anchors
        feature_map_size = [160, 80, 40, 20]
        downsize_factor = [8, 16, 32, 64]
    else:
        anchors = yolov7_anchors
        feature_map_size = [52, 26, 13]
        downsize_factor = [8, 16, 32]

    if opt.onnxrt == 1:
        session = ort.InferenceSession(f'{opt.onnx_model_path}')
        output = [node.name for node in session.get_outputs()]
        img = img.numpy()
        pred = session.run(None, {'images': img})
        model_output = []
        for out in pred:
            model_output.append(torch.from_numpy(out))
            print(out.shape)

    if opt.qaic == 1:
        sys.path.insert(0, "/opt/qti-aic/dev/lib/x86_64/")
        import qaicrt

        qpc = qaicrt.Qpc("aic_binary_fp16")
        output_nodes = []
        for buf in qpc.getBufferMappings():
            output_nodes.append(buf.bufferName)
        output_nodes = output_nodes[1:]

        print('Running Model using qaic-exec....')

        os.system(
            f"qaic-runner -t aic_binary_fp16 -d {opt.device} -i input.raw --write-output-dir=aic_output"
        )
        model_output = []
        for i in range(len(output_nodes)):

            p = np.fromfile(
                f'./aic_output/{output_nodes[i]}-activation-0-inf-0.bin',
                dtype=np.float32)
            p = p.reshape(1, 3, feature_map_size[i], feature_map_size[i], 85)
            p = torch.from_numpy(p)
            model_output.append(p)
    model_decoded_output = []
    for i in range(len(model_output)):
        # Decoding Part
        img_size = list(imgsz)

        decode = YOLOLayer(np.array(anchors[i]), 80, img_size, i, [],
                           downsize_factor[i],
                           [feature_map_size[i], feature_map_size[i]])
        out = decode(model_output[i])
        model_decoded_output.append(out)

    return torch.cat(model_decoded_output, 1)


def detect(save_img=True):
    #decode boxes info
    imgsz = (opt.img_size, opt.img_size)
    filename = os.path.splitext(opt.img_path)[0]
    if os.path.exists(opt.output):
        shutil.rmtree(opt.output)  # delete output folder
    os.makedirs(opt.output)  # make new output folder

    # Get names and colors
    import yaml
    with open(opt.names) as f:
        names = yaml.load(f, Loader=yaml.SafeLoader)['names']
    #names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    #Pre-processing
    im0s = cv2.imread(opt.img_path)
    img = letterbox(im0s, imgsz, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(
        np.float32)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    path = opt.img_path
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    img.tofile("input.raw")
    img = torch.from_numpy(img)

    pred = inference(img, opt)
    pred = non_max_suppression(pred,
                               opt.conf_thres,
                               opt.iou_thres,
                               classes=opt.classes,
                               agnostic=opt.agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections for image i
        p, s, im0 = path, '', im0s

        save_path = str(Path(opt.output) / Path(p).name)
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from imgsz to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                      im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if opt.save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh
                    with open(save_path[:save_path.rfind('.')] + '.txt',
                              'a') as file:
                        file.write(
                            ('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                if save_img or opt.view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(
                        xyxy,
                        im0,
                        label=label,
                        color=colors[int(cls)],
                    )
                    print(xyxy, label)

        if save_img:
            #if dataset.mode == 'images':
            cv2.imwrite(save_path, im0)

    if opt.save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + opt.output)

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--names',
                        type=str,
                        default='../yolov7/data/coco.yaml',
                        help='*.names path')
    parser.add_argument('--agnostic-nms',
                        action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--output',
                        type=str,
                        default='output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size',
                        type=int,
                        default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.25,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.45,
                        help='IOU threshold for NMS')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img',
                        action='store_true',
                        help='display results')
    parser.add_argument('--save-txt',
                        action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--classes',
                        nargs='+',
                        type=int,
                        help='filter by class')
    parser.add_argument('--onnxrt',
                        type=int,
                        default=0,
                        help='onnxrt inference')
    parser.add_argument('--qaic',
                        type=int,
                        default=0,
                        help='qaic-exec inference')
    parser.add_argument("--img-path",
                        required=True,
                        default="../inputFiles/horses.jpg",
                        help="Image path for inference")
    parser.add_argument("--onnx-model-path",
                        required=True,
                        default="ONNX/yolov7_640_640_smartNMS.onnx",
                        help="model name")
    parser.add_argument("--model-name",
                        required=False,
                        default="yolov7",
                        help="model name")
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()