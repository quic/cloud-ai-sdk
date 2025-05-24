# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

image=ghcr.io/quic/cloud_ai_inference_ubuntu22:1.19.8.0
qpc_path=/path/to/qpc

docker run -dit \
  --workdir /model \
  --name qaic-vllm \
  --network host \
  --mount type=bind,source=${PWD}/serve.sh,target=/model/serve.sh \
  --mount type=bind,source=${qpc_path},target=/model/qpc \
  -v qaic-vllm:/model/data \
  --env VLLM_QAIC_MAX_CPU_THREADS=8 \
  --env VLLM_QAIC_QPC_PATH=/model/qpc \
  --env HF_HOME=/model/data/huggingface \
  --env QEFF_HOME=/model/data/qeff_models \
  --device=/dev/accel/accel0 \
  --device=/dev/accel/accel1 \
  --device=/dev/accel/accel2 \
  --device=/dev/accel/accel3 \
  --entrypoint=/model/serve.sh \
  ${image}
