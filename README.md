
<p align="center">
  <picture>
    <img alt="Cloud AI 100" src="images/Cloud_AI_100.png" width=100%>
  </picture>
</p>


<p style="font-size: 20px" align="center">
| <a href="https://quic.github.io/cloud-ai-sdk-pages/latest/"><b>User Guide</b></a> | <a href="https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/"><b>Download SDK</b></a> | <a href="https://quic.github.io/cloud-ai-sdk-pages/latest/blogs/Train_Anywhere/train_anywhere/"><b>Blog</b></a> |

</p>

---

# Qualcomm Cloud AI Developer Resources


---

*Latest News* 🔥
- [2026/04] Added [YOLO Triton server](models/vision/detection) example
- [2026/04] Added [Mask R-CNN](models/vision/segmentation/mask_rcnn) model example using Detectron2 for instance segmentation
- [2026/04] [QAic Bench](utils/qaic-bench) now supports vLLM 0.10, QPC download, and random prompt generation
- [2026/04] [Multi-device](utils/multi-device) utility updated to use switchtec for ACS disable
- [2025/08] Try the [QAic Bench](utils/qaic-bench) script for LLM benchmarking on Cloud AI accelerators
- [2025/08] The [Open WebUI tutorial](tutorials/open-webui) shows how to use Open WebUI's chat interface with Cloud AI accelerators.
- [2025/08] Added [Kubernetes](tutorials/Kubernetes) tutorial
- [2025/08] Added [Efficient Transformers](tutorials/efficient_transformers) tutorial
- [2025/08] Added [DETR ResNet-50](tutorials/Computer-Vision/DETR) model example
- [2025/08] Added [YOLOv8](models/vision/detection) model example
- [2024/11] Check out the [Playground Tutorial](tutorials/Playground) to learn how to access the latest Generative AI models running on Qualcomm Cloud AI 100 Ultra Accelerators hosted in the cloud.
- [2024/11] Added [Stable Diffusion XL Turbo](models/multimodal/text_to_image/sdxl_turbo) model example
- [2024/11] Added [Stable Diffusion 3.5 Medium](models/multimodal/text_to_image/stable-diffusion-3.5-medium) model example
- [2024/09] Added [Whisper](models/speech/whisper) model example
- [2024/09] Added [SDXL-DeepCache](models/multimodal/text_to_image/sdxl_deepcache) model example
- [2024/04] Qualcomm released [efficient transformers](https://github.com/quic/efficient-transformers) for seamless deployment of pre-trained LLMs.
- [2024/03] Added [AI 100 Ultra recipe for Llama](models/language_processing/decoder/LlamaForCausalLM) - e.g., [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b)
- [2024/03] Added support for Speculative Decoding with LLMs - [CodeGen with Speculative Decoding](models/language_processing/decoder/CodeGen-With-Speculative-Decoding)
- [2024/02] Added support for [Stable Diffusion XL](models/multimodal/text_to_image/sdxl_turbo)
- [2024/02] Added support for [MPT family of LLMs](models/language_processing/decoder/MptForCausalLM) - e.g., [MPT-7B](https://huggingface.co/mosaicml/mpt-7b)
- [2024/02] Added support for [GPTBigCode family of LLMs](models/language_processing/decoder/GPTBigCodeForCausalLM) - e.g., [StarCoder](https://huggingface.co/bigcode/starcoder)
- [2024/01] Added [profiling tutorial for LLMs](tutorials/NLP/Profiler-Intermediate/Profiler-LLM.ipynb)
- [2024/01] Added support for [DeciDiffusion-v2.0](models/multimodal/text_to_image/DeciDiffusion-v2-0)
- [2024/01] Added support for [DeciCoder-6B](models/language_processing/decoder/DeciCoder-6b)
- [2024/01] Added support for [Llama family of LLMs](models/language_processing/decoder/LlamaForCausalLM) - e.g., [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b)
---

## About
Qualcomm Cloud AI 100 offers a unique blend of high computational performance, low latency, and low power utilization, making it well-suited for a broad range of AI applications, including computer vision, natural language processing, and Generative AI such as Large Language Models (LLMs). Specifically designed for high-performance, low-power AI processing, it is ideal for both public and private cloud environments, supporting Enterprise AI applications.

This repository provides developers with 3 key resources
- Models    - Recipes for [CV](models/vision), [NLP](models/language_processing), [multimodal](models/multimodal/text_to_image) models to run on Cloud AI platforms performantly, <br> For LLM, embeddings and speech models, see [efficient-transformers](https://github.com/quic/efficient-transformers)
- Tutorials - [Tutorials](tutorials) cover model onboarding, performance tuning, and profiling aspects of inferencing across CV/NLP on Cloud AI platforms
- Samples   - Sample code illustrating usage of APIs - [Python](samples/python) and [C++](samples/cpp/cpp_qpc_inference) for inference on Cloud AI platforms

## 🚀 Quick Start

### 1. YOLOv8 Object Detection (Docker)

```bash
# Build Docker image
cd models/vision/detection
docker compose build

# Export YOLO model and generate Triton repo
docker compose run --rm yolo-export

# Start Triton server
docker compose up yolo-triton

# Run inference client
pip3 install numpy Pillow
python3 triton_client.py --endpoint localhost:8000

# Check results in output.jpg
```

See [YOLO Detection](models/vision/detection/) for more model options.

### 2. LLM with vLLM Container

```bash
docker run --rm -it --network host \
   --device /dev/accel/ \
   --shm-size=2gb \
   --mount type=bind,source=$HOME/.cache,target=/cache \
   -e HF_HOME=/cache/huggingface \
   -e QEFF_HOME=/cache/qeff_models \
   ghcr.io/quic/cloud_ai_inference_vllm:1.21.4.0 \
   --host 127.0.0.1 \
   --port 8080 \
   --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
   --max-model-len 8192 \
   --max-num-seq 1 \
   --max-seq-len-to-capture 128 \
   --quantization mxfp6 \
   --kv-cache-dtype mxint8
```

See [vLLM for QAic User Guide](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Model-Serving/vLLM-Serving) for vLLM usage.

After starting the server, test with curl:
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

## 📚 Supported Models

### 💾 Pre-compiled Model Catalog 
For pre-compiled model binaries refer to the [Model Catalog](https://dc00tk1pxen80.cloudfront.net/QPC/catalog-index) (independent partner site)

### 💬 Generative AI - Large Language Models (LLM), Vision Language Models (VLM), Embeddings, Speech, Image Generation, Video Generation
- See [efficient-transformers](https://github.com/quic/efficient-transformers)

### 🤖 [NLP - Encoder-only Transformer Models](models/language_processing/encoder)
- 80+ models including all varieties of `bert` models, `sentence-transformer` embedding models, etc.

### 👀 [Computer Vision (CV) Models](models/vision/)
- ViT (`vit_b_16`, `vit_b_32`, `vit-base-patch16-224`)
- YOLO (`yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`, `yolov7-e6e`, `yolov8m`)
- Mask R-CNN (Detectron2-based instance segmentation with bounding boxes and masks)
- ResNet (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`)
- ResNeXt (`resnext101_32x8d`, `resnext101_64x4d`, `resnext50_32x4d`)
- Wide ResNet (`wide_resnet101_2`, `wide_resnet50_2`)
- DenseNet (`densenet121`, `densenet161`, `densenet169`, `densenet201`)
- MNASNet (`mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, `mnasnet1_3`)
- MobileNet (`mobilenet_v2`, `mobilenet_v3_large`, `mobilenet_v3_small`)
- EfficientNet (`efficientnet_v2_l`, `efficientnet_v2_m`, `efficientnet_v2_s`, `efficientnet_b0`, `efficientnet_b7`, etc.)
- ShuffleNet (`shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`)
- SqueezeNet (`squeezenet1_0`, `squeezenet1_1`)

## Support
Reach out on the [📢cloud-ai Discord channel](https://discord.com/invite/qualcommdevelopernetwork) or use [💬 GitHub Issues](https://github.com/quic/cloud-ai-sdk/issues) to request for model support, raise questions or to provide feedback.

## Disclaimer
While this repository may provide documentation on how to run models on Qualcomm Cloud AI platforms, this repository does NOT contain any of these models.  All models referenced in this documentation are independently provided by third parties at unaffiliated websites. Please be sure to review any third-party license terms at these websites; no license to any model is provided in this repository. This repository of documentation provides no warranty or assurances for any model so please also be sure to review all model cards, model descriptions, model limitations / intended uses, training data, biases, risks, and any other warnings given by the third party  model providers.

## License
The documentation made available in this repository is licensed under the BSD 3-clause-Clear “New” or “Revised” License. Check out the [LICENSE](LICENSE) for more details.
