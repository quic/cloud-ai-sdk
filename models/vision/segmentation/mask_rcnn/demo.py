####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import argparse

from PIL import Image, ImageDraw, ImageFont

from qaic_models.vision.segmentation.mask_rcnn.model import MaskRCNN
from qaic_models.common.postprocess import COCO_CATEGORIES

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to an RGB image")
    ap.add_argument(
        "--model",
        required=True,
        help="Directory containing model_backbone_fpn.onnx, model_rpn.onnx, model_roi_heads.onnx",
    )
    ap.add_argument("--top", type=int, default=20, help="How many boxes to print")
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--nms-thresh", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="Optional output image path with drawn detections")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup iterations for each ONNX session")
    ap.add_argument("--runs", type=int, default=10, help="Timed iterations for each ONNX session")
    args = ap.parse_args()

    # Run local QAIC split-ONNX pipeline
    provider = "QAicExecutionProvider"
    model = MaskRCNN(args.model, provider=provider)
    res = model.predict(
        args.image,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        topk=args.top,
        warmup=args.warmup,
        runs=args.runs,
    )

    for name, st in res.latencies.items():
        print(
            f"latency {name}: mean={st.mean_ms:.3f}ms p50={st.p50_ms:.3f}ms "
            f"p90={st.p90_ms:.3f}ms p99={st.p99_ms:.3f}ms"
        )

    pp = res.post

    # Print + draw boxes + masks
    print(f"num detections (after postprocess): {pp.boxes_xyxy.shape[0]}")
    print("idx\tscore\tclass\tx1\ty1\tx2\ty2")

    vis_img = Image.open(args.image).convert("RGB").resize((1216, 800), resample=Image.BILINEAR)

    # If masks are available, create an overlay and shade instances
    if getattr(pp, "masks", None) is not None and pp.masks is not None:
        import numpy as np

        base = np.array(vis_img).astype(np.float32)
        overlay = base.copy()
        alpha = 0.5

        for j in range(pp.boxes_xyxy.shape[0]):
            mask = pp.masks[j]
            if mask is None:
                continue
            cls_id = int(pp.classes[j])
            # Simple class-dependent color (repeat every few classes)
            color_idx = cls_id % 7
            colors = [
                (255, 0, 0),    # red
                (0, 255, 0),    # green
                (0, 0, 255),    # blue
                (255, 255, 0),  # yellow
                (255, 0, 255),  # magenta
                (0, 255, 255),  # cyan
                (255, 127, 0),  # orange
            ]
            color = np.array(colors[color_idx], dtype=np.float32)

            m = mask.astype(bool)
            if m.shape != base.shape[:2]:
                # If mask is not full-image size, skip resizing here; this implementation
                # assumes masks are already in image coordinates.
                pass

            overlay[m] = (1 - alpha) * overlay[m] + alpha * color

        vis_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(vis_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for j in range(pp.boxes_xyxy.shape[0]):
        b = pp.boxes_xyxy[j]
        score = float(pp.scores[j])
        cls_id = int(pp.classes[j])

        x1, y1, x2, y2 = map(float, b.tolist())
        print(f"{j}\t{score:.4f}\t{cls_id}\t{x1:.1f}\t{y1:.1f}\t{x2:.1f}\t{y2:.1f}")

        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # label background
        cls_name = COCO_CATEGORIES[cls_id] if 0 <= cls_id < len(COCO_CATEGORIES) else str(cls_id)
        label = f"{cls_name}:{score:.2f}"
        if font is not None:
            try:
                bbox = draw.textbbox((x1, y1), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = draw.textsize(label, font=font)
        else:
            tw, th = (len(label) * 6, 10)

        tx1, ty1 = x1, max(0.0, y1 - th - 2)
        tx2, ty2 = x1 + tw + 4, ty1 + th + 2
        draw.rectangle([tx1, ty1, tx2, ty2], fill="red")
        draw.text((tx1 + 2, ty1 + 1), label, fill="white", font=font)

    if args.out:
        vis_img.save(args.out)
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()

