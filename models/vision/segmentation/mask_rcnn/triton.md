# Triton Inference Server Instructions

## Triton Docker Image

Before serving with Triton Inference Server, run this step once to generate the Triton Docker Image `qaic-triton:latest` with mask_rcnn dependencies pre-installed.

To run `docker` commands, add yourself to the [docker](https://docs.docker.com/engine/install/linux-postinstall/) group, or add `sudo` to the commands.

Launch Triton container:

```bash
docker run -dit \
  --shm-size=4g \
  --name qaic-triton \
  --network host \
  --mount type=bind,source=</path/to/cloud-ai-sdk root>,target=/cloud-ai-sdk \
  --device /dev/accel/ \
  ghcr.io/quic/cloud_ai_triton_server:1.21.2.0

docker exec -it qaic-triton bash

```

Environment setup:

```bash
cd /cloud-ai-sdk
pip3 install ".[mask-rcnn-infer]"
```

Exit container:

```bash
exit
```

Commit container:

```bash
docker commit qaic-triton qaic-triton:latest
```

Remove container:

```bash
docker stop qaic-triton && docker rm qaic-triton
```

## Generate Model Repository

First, complete the setup steps in [README](README.md).

Generate a Triton model repository for the model:

```bash
python3 -m qaic_models.vision.segmentation.mask_rcnn.export --triton-model-repo ./triton-model-repo
```

## Launch Triton Container

Start Triton server:

```bash
docker run -it --rm \
  --shm-size=4g \
  --network host \
  --mount type=bind,source=</path/to/cloud-ai-sdk root>,target=/cloud-ai-sdk \
  --device /dev/accel/ \
  qaic-triton:latest \
  /opt/tritonserver/bin/tritonserver --model-repository=/cloud-ai-sdk/models/vision/segmentation/mask_rcnn/triton-model-repo
```

## Run the Triton Model

Dependencies:

First, complete the setup steps in [README](README.md).

Install additional Triton packages:

```bash
pip3 install tritonclient gevent geventhttpclient
```

Sample code:

```python
import numpy as np
import tritonclient.http as triton_http
from tritonclient.http import InferInput, InferRequestedOutput

from qaic_models.vision.segmentation.mask_rcnn.preprocess import MaskRCNNImage

TRITON_URL = "localhost:8000"
MODEL_NAME = "detectron2_maskrcnn_ensemble_qaic"

def main():
    image_path = "000000000139.jpg"  # change as needed

    # Preprocess using MaskRCNNImage: returns NCHW RGB float32 in model's expected range
    # size=(width, height)
    img_nchw = MaskRCNNImage.load(image_path, size=(1216, 800), batch=1)
    # Ensure contiguous
    img_nchw = np.ascontiguousarray(img_nchw.astype(np.float32))

    # Create Triton HTTP client
    client = triton_http.InferenceServerClient(url=TRITON_URL, verbose=False)

    # Prepare Triton input
    infer_input = InferInput("image", img_nchw.shape, "FP32")
    infer_input.set_data_from_numpy(img_nchw)

    # Request ensemble outputs
    outputs = [
        InferRequestedOutput("cls_results"),
        InferRequestedOutput("cls_counts"),
        InferRequestedOutput("boxes"),
    ]

    # Send inference request
    result = client.infer(
        model_name=MODEL_NAME,
        inputs=[infer_input],
        outputs=outputs,
    )

    # Fetch outputs
    cls_results = result.as_numpy("cls_results")  # [N, 2] -> [class_id, score]
    cls_counts = result.as_numpy("cls_counts")    # [1] or [B,1]
    boxes = result.as_numpy("boxes")              # [N, 4]

    print("cls_counts:", cls_counts)
    print("cls_results shape:", cls_results.shape)
    print("boxes shape:", boxes.shape)

    # Print a few detections
    num = int(cls_counts[0]) if cls_counts is not None else cls_results.shape[0]
    for i in range(min(num, 5)):
        cls_id = int(cls_results[i, 0])
        score = float(cls_results[i, 1])
        x1, y1, x2, y2 = boxes[i]
        print(f"{i}: cls={cls_id} score={score:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

if __name__ == "__main__":
    main()
```
