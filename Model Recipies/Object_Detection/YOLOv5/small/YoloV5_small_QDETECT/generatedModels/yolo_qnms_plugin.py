###############################################################################
# Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
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
###############################################################################

import torch
import torch.utils.cpp_extension
from torch.onnx.symbolic_helper import parse_args

op_source = """
#include <torch/script.h>
#include <iostream>
#include <queue>
#include <assert.h>


using dim_t = int64_t;
using ClassBox = std::pair<float, dim_t>;

static void maxMin(float lhs, float rhs, float &min, float &max) {
  if (lhs >= rhs) {
    min = rhs;
    max = lhs;
  } else {
    min = lhs;
    max = rhs;
  }
}

enum class NMSType : int { COMBINED };

struct CombinedNMSBox {
  float score;
  size_t classIdx;
  std::array<float, 4> coords;
};

static bool doIOU(float *boxes, dim_t batchIndex,
                  dim_t selectedBoxIndex, dim_t candidateBoxIndex,
                  int centerPointBox, float iouThreshold, dim_t numBoxes, 
                  dim_t Q, dim_t numCords, NMSType nmsType, dim_t classIdx = 0) {
  float sx[] = {0.0f, 0.0f, 0.0f, 0.0f};
  float cx[] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (nmsType == NMSType::COMBINED) {
    for (dim_t i = 0; i < 4; ++i) {
      sx[i] = boxes[((batchIndex * numBoxes + selectedBoxIndex) * Q + classIdx) * numCords + i];
      cx[i] = boxes[((batchIndex * numBoxes + candidateBoxIndex) * Q + classIdx) * numCords + i];
    }
  }
  float xSMin = 0.0f;
  float ySMin = 0.0f;
  float xSMax = 0.0f;
  float ySMax = 0.0f;
  float xCMin = 0.0f;
  float yCMin = 0.0f;
  float xCMax = 0.0f;
  float yCMax = 0.0f;
  // Standardizing coordinates so that (xmin, ymin) is upper left corner of a
  // box and (xmax, ymax) is lower right corner of the box.
  if (!centerPointBox) {
    // 0 means coordinates for diagonal ends of a box.
    // Coordinates can either be absolute or normalized.
    maxMin(sx[0], sx[2], xSMin, xSMax);
    maxMin(sx[1], sx[3], ySMin, ySMax);
    maxMin(cx[0], cx[2], xCMin, xCMax);
    maxMin(cx[1], cx[3], yCMin, yCMax);
  } else {
    float halfWidthS = sx[2] / 2.0f;
    float halfHeightS = sx[3] / 2.0f;
    float halfWidthC = cx[2] / 2.0f;
    float halfHeightC = cx[3] / 2.0f;
    xSMin = sx[0] - halfWidthS;
    ySMin = sx[1] - halfHeightS;
    xSMax = sx[0] + halfWidthS;
    ySMax = sx[1] + halfHeightS;
    xCMin = cx[0] - halfWidthC;
    yCMin = cx[1] - halfHeightC;
    xCMax = cx[0] + halfWidthC;
    yCMax = cx[1] + halfHeightC;
  }
  // finding upper left and lower right corner of a box formed by intersection.
  float xMin = std::max(xSMin, xCMin);
  float yMin = std::max(ySMin, yCMin);
  float xMax = std::min(xSMax, xCMax);
  float yMax = std::min(ySMax, yCMax);
  float intersectionArea =
      std::max(0.0f, xMax - xMin) * std::max(0.0f, yMax - yMin);
  if (intersectionArea == 0.0f) {
    return false;
  }
  float sArea = (xSMax - xSMin) * (ySMax - ySMin);
  float cArea = (xCMax - xCMin) * (yCMax - yCMin);
  float unionArea = sArea + cArea - intersectionArea;
  return intersectionArea > iouThreshold * unionArea;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    CustomQnmsYolo(torch::Tensor boxes, torch::Tensor scores,
                   torch::Tensor max_output_boxes_per_class,
                   torch::Tensor max_total_size_quantity, torch::Tensor iou_thres,
                   torch::Tensor score_thres, int64_t clip_boxes, int64_t pad_per_class) {
    // boxes    : [B x N x 1 x 4]
    constexpr int boxesBatchDim = 0;
    constexpr int boxesBoxDim = 1;
    constexpr int boxesQDim = 2;
    constexpr int boxesCordDim = 3;
    float *boxesData = (float *)boxes.data_ptr<float>();
    // scores   : [B x N x 80]
    constexpr int scoresBatchDim = 0;
    constexpr int scoresBoxDim = 1;
    constexpr int scoresClassDim = 2;
    float *scoresData = (float *)scores.data_ptr<float>();
    const auto numBatches = scores.sizes()[scoresBatchDim];
    const auto numClasses = scores.sizes()[scoresClassDim];
    const auto numBoxes = boxes.sizes()[boxesBoxDim];
    const auto Q = boxes.sizes()[boxesQDim];
    dim_t numCords = boxes.sizes()[boxesCordDim];

    assert(boxes.sizes()[boxesBoxDim] == scores.sizes()[scoresBoxDim] &&
            "Mismatch between number of scores and number of boxes.");
    assert(boxes.sizes()[boxesBatchDim] == scores.sizes()[scoresBatchDim] &&
            "Mismatch in batch dimension.");
    assert(boxes.sizes()[boxesCordDim] == 4 &&
            "Boxes should have exact 4 coordinates only.");

    (void)boxesBatchDim;
    (void)scoresBoxDim;
    (void)boxesBoxDim;
    constexpr unsigned centerPointBox = 0;
    auto cmpFunc = [](const ClassBox &a, const ClassBox &b) {
        if (a.first == b.first) {
        return a.second > b.second;
        }
        return a.first < b.first;
    };
    std::vector<ClassBox> selectedIndices(numBoxes);
    std::vector<std::vector<CombinedNMSBox>> outBoxes(numBatches);
    float score_threshold = score_thres.item<float>();
    float iouThreshold = iou_thres.item<float>();
    dim_t maxOutputSizePerClass = max_output_boxes_per_class.item<int64_t>();
    dim_t maxTotalSize = max_total_size_quantity.item<int64_t>();
    // The CustomQnmsYolo module will output "max" number of bounding boxes at most per an image.
    dim_t maxBoxes = max_total_size_quantity.item<int64_t>();
    torch::Tensor nmsedBoxes = torch::zeros({numBatches, maxBoxes, numCords}, torch::kF32);
    torch::Tensor nmsedScores = torch::zeros({numBatches, maxBoxes}, torch::kF32);
    torch::Tensor nmsedClasses = torch::zeros({numBatches, maxBoxes}, at::dtype(at::kInt));
    torch::Tensor validDetections = torch::zeros({numBatches}, at::dtype(at::kInt));
    float *nmsedBoxesH = (float *)nmsedBoxes.data_ptr<float>();
    float *nmsedScoresH = (float *)nmsedScores.data_ptr<float>();
    int32_t *nmsedClassesH = (int32_t *)nmsedClasses.data_ptr<int32_t>();
    int32_t *validDetectionsH = (int32_t *)validDetections.data_ptr<int32_t>();
    for (dim_t batchIndex = 0; batchIndex < numBatches; ++batchIndex) {
        dim_t outPutBoxIndex = 0;
        int32_t detectedPerBatch = 0;
        for (dim_t classIndex = 0; classIndex < numClasses; ++classIndex) {
            selectedIndices.clear();
            size_t detectedPerClass = 0;
            std::priority_queue<ClassBox, std::vector<ClassBox>, decltype(cmpFunc)>
                queue(cmpFunc);
            for (size_t boxIndex = 0; boxIndex < numBoxes; ++boxIndex) {
                float classValue = scoresData[(batchIndex * numBoxes + boxIndex) *
                                    numClasses + classIndex];
                if (classValue > score_threshold) {
                    queue.emplace(classValue, boxIndex);
                }
            }
            while (!queue.empty()) {
                auto priorBox = queue.top();
                queue.pop();
                bool selected = true;
                for (auto &sBox : selectedIndices) {
                    if (doIOU(boxesData, batchIndex, sBox.second, priorBox.second,
                                centerPointBox, iouThreshold, numBoxes, Q, numCords, 
                                NMSType::COMBINED, Q > 1 ? classIndex : 0)) {
                        selected = false;
                        break;
                    }
                }
                if (selected) {
                    selectedIndices.emplace_back(priorBox);
                    const auto score =
                        scoresData[(batchIndex * numBoxes + priorBox.second) * numClasses + classIndex];
                    std::array<float, 4> boxCoords;
                    for (dim_t i = 0; i < 4; ++i) {
                        auto clsIdx = Q > 1 ? classIndex : 0;
                        boxCoords[i] = boxesData[
                            ((batchIndex * numBoxes + priorBox.second) * Q + clsIdx) * numCords + i];
                    }
                    outBoxes[batchIndex].push_back({score, classIndex, boxCoords});
                    ++outPutBoxIndex;
                    ++detectedPerClass;
                    ++detectedPerBatch;
                }
                if (maxOutputSizePerClass == detectedPerClass) {
                    break;
                }
            }
        }
        validDetectionsH[batchIndex] =
        detectedPerBatch > maxTotalSize ? maxTotalSize : detectedPerBatch;
    }
    auto clipCoord = [](auto val, auto low, auto high) {
        return std::max(low, std::min(val, high));
    };
    for (size_t batch = 0; batch < numBatches; ++batch) {
        dim_t boxIndex = 0;
        std::stable_sort(
            outBoxes[batch].begin(), outBoxes[batch].end(),
            [](auto &lhs, auto &rhs) { return lhs.score > rhs.score; });
        for (const auto &box : outBoxes[batch]) {
            if (boxIndex == maxTotalSize)
                break;
            for (dim_t i = 0; i < 4; ++i) {
                auto val = box.coords[i];
                if (clip_boxes) {
                    val = clipCoord(val, 0.f, 1.f);
                }
                nmsedBoxesH[(batch * maxBoxes + boxIndex) * numCords + i] = val;
            }
            nmsedClassesH[batch * maxBoxes + boxIndex] = box.classIdx;
            nmsedScoresH[batch * maxBoxes + boxIndex] = box.score;
            ++boxIndex;
        }
    }
    return std::make_tuple(
           nmsedBoxes,
           nmsedScores,
           nmsedClasses,
           validDetections );
    // detection_boxes      : [1 x 100 x 4] : float32
    // detection_scores     : [1 x 100] : float32
    // detection_classes    : [1 x 100] : int32
    // valid_detection      : [1] : int32
}

TORCH_LIBRARY(QAic, m) {
  m.def("CustomQnmsYolo", &CustomQnmsYolo);
}
"""
# Compile and load the custom op
torch.utils.cpp_extension.load_inline(
    name="CustomQnmsYolo",
    cpp_sources=op_source,
    is_python_module=False,
    verbose=True,
)
# Wrapper module for custom relu C++ op
class CustomQnmsYolo(torch.nn.Module):
    def __init__(self, clip_boxes, pad_per_class):
        super().__init__()
        self.pad_per_class = torch.tensor([pad_per_class], dtype=torch.int32)
        self.clip_boxes = torch.tensor([clip_boxes], dtype=torch.int32)
        # The class member variables will become attribute of CustomQnmsYolo 
        # Node in the final graph
    def forward(self, boxes, scores, \
                    max_output_boxes_per_class, \
                    max_total_size_quantity, \
                    iou_thres, score_thres):
        score_thres = torch.tensor([score_thres])
        iou_thres = torch.tensor([iou_thres])
        max_output_boxes_per_class = torch.tensor(\
                                        [max_output_boxes_per_class], \
                                        dtype=torch.int32)
        max_total_size_quantity = torch.tensor(\
                                        [max_total_size_quantity], \
                                        dtype=torch.int32)
        # These will be the input for the CustomQnmsYolo Node in the final graph
        # boxes : [B x N x 1 x 4]
        # scores : [B x N x 80]
        return torch.ops.QAic.CustomQnmsYolo(boxes, scores, \
                            max_output_boxes_per_class, \
                            max_total_size_quantity, \
                            iou_thres, score_thres, \
                            self.clip_boxes, self.pad_per_class)
# ONNX export symbolic helper
@parse_args('v', 'v', 'v', 'v', 'v', 'v', 'i', 'i')
# Here for each input being passed to CustomQnmsYolo custom plugin we need to
# explicitly provide arg type.
# More detail on which arg to be used can be found at below link.
# Ref. : https://github.com/pytorch/pytorch/master/torch/onnx/symbolic_helper.py#L128
def CustomQnmsYolo_fn(g, boxes, scores, \
                    max_output_boxes_per_class, \
                    max_total_size_quantity, \
                    iou_thres, score_thres, \
                    clip_boxes, pad_per_class):
    return g.op("QAic::CustomQnmsYolo", boxes, scores, \
                            max_output_boxes_per_class, \
                            max_total_size_quantity, \
                            iou_thres, \
                            score_thres, \
                            clip_boxes_i=clip_boxes, \
                            pad_per_class_i=pad_per_class, \
                            outputs=4)
    # Here, as we will get 4 outputs, we should explicitly mention the same.
    # Also, for clip_boxes and pad_per_class variables, as they as class member
    # variables which will be treated as attributes in the final graph. For 
    # these clip_boxes_i and pad_per_class_i arguments should be used 
    # respectively. Here used i as these are having int32 datatype.

torch.onnx.register_custom_op_symbolic('QAic::CustomQnmsYolo', \
                                        CustomQnmsYolo_fn, 11)

if __name__ == "__main__":
    model = CustomQnmsYolo(clip_boxes=0, pad_per_class=0)
    model.eval()

    torch.manual_seed(42)
    
    boxes = torch.rand(1, 25400, 1, 4).to(torch.float32)
    scores = torch.rand(1, 25400, 80).to(torch.float32)
    
    max_boxes_per_class = torch.tensor(26, dtype=torch.int32)
    max_total_size = torch.tensor(100, dtype=torch.int32)
    iou_threshold = torch.tensor(0.65, dtype=torch.float32)
    score_threshold = torch.tensor(0.001, dtype=torch.float32)
    
    output = model(boxes=boxes, scores=scores, \
                    max_output_boxes_per_class=max_boxes_per_class, \
                    max_total_size_quantity=max_total_size, \
                    iou_thres=iou_threshold, score_thres=score_threshold)
    
    res_boxes, res_scores, res_classes, res_num_boxes = output
    
    # res_boxes.detach().numpy().tofile(\
    #                       "./reference/detection_boxes_out_0_0.raw")
    # res_scores.detach().numpy().tofile(\
    #                       "./reference/detection_scores_out_0_0.raw")
    # res_classes.detach().numpy().tofile(\
    #                       "./reference/detection_classes_out_0_0.raw")
    # res_num_boxes.detach().numpy().tofile(\
    #                       "./reference/valid_detections_out_0_0.raw")
    
    traced = torch.jit.trace(model, (boxes, scores, max_boxes_per_class, \
                                    max_total_size, iou_threshold, \
                                    score_threshold))
    traced.save("customQnmsYolo.pt")
    
    torch.onnx.export(model, (boxes, scores, max_boxes_per_class, \
                            max_total_size, iou_threshold, \
                            score_threshold), \
                            'customQnmsYolo.onnx', \
                            input_names=["boxes", "scores", \
                                        "max_output_size_per_class", \
                                        "max_total_size", \
                                        "iou_threshold", "score_threshold"], \
                            output_names=["detection_boxes", \
                                          "detection_scores", \
                                          "detection_classes", \
                                          "valid_detections"], \
                            opset_version=11)
    
    # boxes.detach().numpy().tofile("./boxes.raw")
    # scores.detach().numpy().tofile("./scores.raw")
    # max_boxes_per_class.detach().numpy().tofile("./max_boxes_per_class.raw")
    # max_total_size.detach().numpy().tofile("./max_total_size.raw")
    # iou_threshold.detach().numpy().tofile("./iou_threshold.raw")
    # score_threshold.detach().numpy().tofile("./score_threshold.raw")
    
    print("Sample CustomQnmsYolo Pt and Onnx model saved successfully")