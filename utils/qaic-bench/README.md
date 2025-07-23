# qaic-bench

Benchmarking script for Cloud AI Inference accelerators.

## Installation for x86_64

Download Cloud AI Docker Image:

```
docker pull ghcr.io/quic/cloud_ai_inference_ubuntu22:1.19.8.0
```

Start container. This example maps 4 Cloud AI 100 Ultra Accelerators. Each accelerator has 4 SoC devices.

Note: For QPC generation, choose a `/cache` location with 1TB or more of free space to hold model weights, ONNX files, and QPC model binaries.

Note: Run `docker container rm qaic-bench` to clean up after exiting the container.

```
cd utils/qaic-bench

docker run -it \
  --workdir /app \
  --name qaic-bench \
  --network host \
  --mount type=bind,source=${PWD},target=/app \
  --mount type=bind,source=${HOME}/.cache,target=/cache \
  --env HF_HOME='/cache/huggingface' \
  --env QEFF_HOME='/cache/qeff_models' \
  --env XDG_CACHE_HOME='/cache' \
  --device=/dev/accel/accel0 \
  --device=/dev/accel/accel1 \
  --device=/dev/accel/accel2 \
  --device=/dev/accel/accel3 \
  --device=/dev/accel/accel4 \
  --device=/dev/accel/accel5 \
  --device=/dev/accel/accel6 \
  --device=/dev/accel/accel7 \
  --device=/dev/accel/accel8 \
  --device=/dev/accel/accel8 \
  --device=/dev/accel/accel9 \
  --device=/dev/accel/accel10 \
  --device=/dev/accel/accel11 \
  --device=/dev/accel/accel12 \
  --device=/dev/accel/accel13 \
  --device=/dev/accel/accel14 \
  --device=/dev/accel/accel15 \
  ghcr.io/quic/cloud_ai_inference_ubuntu22:1.19.8.0
```

Activate vLLM environment:<br>

```
source /opt/vllm-env/bin/activate
```

## Installation for AArch64

Follow instructions [here](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/vLLM/vLLM/index.html#installing-from-source) to setup the vLLM environment for Cloud AI from source for AArch64.

Activate vLLM environment:<br>

```
source qaic-vllm-venv/bin/activate
```

## KV-Heads Replication

Download KV-Heads Replication script from Efficient Transformers. This is needed to efficiently tensor-slice large models across 16 SoCs.

```
wget https://github.com/quic/efficient-transformers/raw/refs/heads/release/v1.19.3_fp8_update/scripts/replicate_kv_head/replicate_kv_heads.py
```

## Usage

Example:

```
python3 qaic_bench.py config/config_llama_3_1_8b.json
```

Details:

```
usage: qaic_bench.py [-h] [--devices DEVICES] [--compile-only] config

positional arguments:
  config             JSON file with model configurations

options:
  -h, --help         show this help message and exit
  --devices DEVICES  List of comma separated device IDs to use for inferencing
  --compile-only     Generate QPCs and skip benchmarking
```

## Configuration

### Example

```
{
    "vllm_root": "/opt/qti-aic/integrations/vllm",
    
    "models": [
        {
            "name": "Meta-Llama-3.1-8B-Instruct",
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "configs": [
                {
                    "batch_size": 1,
                    "devices": 4,
                    "prompt_len": 4096,
                    "generation_len": 4096
                }
            ]
        }
    ]
}
```

### JSON Reference

| Property       | Description                               |
| -------------- | ----------------------------------------- |
| vllm_root      | Path to vLLM installation            |
| models         | List of models to benchmark               |

### Model Properties

| Property       | Description                               |
| -------------- | ----------------------------------------- |
| name           | Model friendly name                       |
| model          | Hugging Face model path                   |
| configs        | List of model configurations to benchmark |

### Config Properties

| Property       | Description                               |
| -------------- | ----------------------------------------- |
| batch_size     | Model batch size                          |
| devices        | Number of Cloud AI SoCs for tensor-sliced execution.  Set to 1 for single-SoC execution. |
| prompt_len     | Prompt input length                       |
| generation_len | Max number of output tokens to generate   |
| qpc (optional) | Path to pre-generated QPC binary. If not specified, QPC will be generated. |
