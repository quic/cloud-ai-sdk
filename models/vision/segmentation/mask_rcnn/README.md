# Detectron2 Mask R-CNN

[Detectron2](https://github.com/facebookresearch/detectron2)'s Mask R-CNN is an instance segmentation model built by Facebook AI Research.

These instructions demonstrate how to compile and run Detectron2 Mask R-CNN on Qualcomm Cloud AI accelerators.

## Setup

```bash
sudo apt update && sudo apt install -y build-essential cmake protobuf-compiler libprotobuf-dev

python3.10 -m venv env_maskrcnn
source env_maskrcnn/bin/activate
pip3 install pip -U
pushd </path/to/cloud-ai-sdk root>
pip3 install ".[mask-rcnn]"
popd
```

For an inference-only environment use:

```
pip3 install ".[mask-rcnn-infer]"
```

## Export

Export ONNX and QPC model binaries

```bash
python3 -m qaic_models.vision.segmentation.mask_rcnn.export
```

To limit the maximum number of NSP cores used for model compilation, add the `--max-cores` flag:

```bash
python3 -m qaic_models.vision.segmentation.mask_rcnn.export --max-cores 8
```

To export quantized QPC model binaries, add the `--quantize` flag:

```bash
python3 -m qaic_models.vision.segmentation.mask_rcnn.export --quantize
```

The exporter uses Detectron2 and applies a patch for Cloud AI devices. The patch makes Mask R-CNN export and deployment more deterministic and accelerator-friendly: it forces a fixed inference resolution (800-1216) and fixed RPN Top-K (200), and adds ONNX-export specific code paths that avoid dynamic/shape-dependent behavior (e.g., boolean masking/NonZero indexing and shape-dependent Top-K selection) by replacing them with static pooling and fixed-size logic under torch.onnx.is_in_onnx_export(). The patch adjusts ROIAlign sampling ratios for stability, and adds a split-ONNX export mode (backbone/FPN, RPN, ROI heads) plus an optional FP16 weight clamp to reduce underflow issues.

The patch is currently limited to batchsize=1.

The following ONNX models are generated:

* model_backbone_fpn.onnx (backbone+FPN)
* model_rpn.onnx (RPN)
* model_roi_heads.onnx (ROI)

QPC model binaries are generated with the `qaic-compiler` compiler. Each Cloud AI device contains multiple Neural Signal Processing (NSP) AI Cores. The backbone network is compiled for 12 cores. The ROI region of interest network is compiled for 4 cores.  During inferencing we can run multiple copies of the network in parallel.

## Inferencing

Next we run the compiled model binaries on device.

backbone+FPN runs on QAic for acceleration (it is the dominant compute), the RPN runs on CPU via ONNXRuntime due to its high scratch memory footprint and the ROI heads run on QAic to accelerate per-proposal classification/regression/mask computation. Final box decoding/NMS/postprocessing is also kept on CPU in Python.

`demo.py` shows how to run the split-model pipeline.

```bash
python3 -m qaic_models.vision.segmentation.mask_rcnn.demo --image 000000000139.jpg --model ./ --out out.jpg
```

Bounding boxes and scores are decoded from the model output and used to annotate the image. The result is stored in `out.jpg`.

## Python Inferface

```python
from qaic_models.vision.segmentation.mask_rcnn.model import MaskRCNN

def main() -> None:
    # Update these paths for your environment
    model_dir = "./"  # contains QPCs + model_rpn.onnx
    image_path = "./000000000139.jpg"

    m = MaskRCNN(model_dir, provider="QAicExecutionProvider")

    result = m.predict(
        image_path,
        score_thresh=0.5,
        nms_thresh=0.5,
        topk=100,
        warmup=1,
        runs=1,
    )

    print("Num detections:", len(result.post.boxes_xyxy))
    print("First 5 boxes:", result.post.boxes_xyxy[:5])
    print("First 5 scores:", result.post.scores[:5])
    print("First 5 classes:", result.post.classes[:5])

    print("\nLatencies (ms):")
    for k, v in result.latencies.items():
        print(f"  {k}: mean={v.mean_ms:.3f} (p50={v.p50_ms:.3f}, p90={v.p90_ms:.3f}, p99={v.p99_ms:.3f})")


if __name__ == "__main__":
    main()

```
