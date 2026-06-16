# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""YOLO post-processing with Non-Maximum Suppression."""

import numpy as np

# COCO class names (common classes)
COCO_CLASSES = {
    0: "person", 2: "car", 27: "tie", 15: "cat", 16: "dog",
    17: "horse", 22: "zebra", 23: "giraffe", 41: "cup", 39: "bottle"
}


def compute_iou(box1, box2):
    """Compute Intersection over Union between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    if inter_w == 0 or inter_h == 0:
        return 0.0

    inter_area = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0


def non_max_suppression(detections, iou_thresh=0.45):
    """Apply NMS to remove overlapping detections."""
    if not detections:
        return []

    # Group by class
    by_class = {}
    for det in detections:
        cid = det["class_id"]
        if cid not in by_class:
            by_class[cid] = []
        by_class[cid].append(det)

    result = []
    for cid, dets in by_class.items():
        # Sort by score
        dets.sort(key=lambda x: x["score"], reverse=True)

        keep = []
        while dets:
            best = dets.pop(0)
            keep.append(best)

            # Remove boxes with high IoU
            filtered = []
            for det in dets:
                iou = compute_iou(best["bbox"], det["bbox"])
                if iou <= iou_thresh:
                    filtered.append(det)
            dets = filtered

        result.extend(keep)

    return result


def postprocess_yolo(output, conf_thresh=0.25, iou_thresh=0.45):
    """Parse YOLOv8 output: [84, 8400] -> detections with NMS."""
    output = output.reshape(84, 8400).T  # [8400, 84]
    boxes = output[:, :4]  # x, y, w, h
    scores = output[:, 4:]  # class scores
    max_scores = scores.max(axis=1)
    max_classes = scores.argmax(axis=1)
    mask = max_scores > conf_thresh
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    max_classes = max_classes[mask]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    detections = []
    for i in range(len(boxes)):
        detections.append({
            "class_id": int(max_classes[i]),
            "score": float(max_scores[i]),
            "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])]
        })
    # Apply NMS
    detections = non_max_suppression(detections, iou_thresh)
    detections.sort(key=lambda x: x["score"], reverse=True)
    return detections
