# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

image=ghcr.io/open-webui/open-webui:main

docker run \
  -d \
  --network host \
  -e OPENAI_API_KEY=test-key \
  -e OPENAI_API_BASE_URL="http://localhost:8000/v1" \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ${image}
