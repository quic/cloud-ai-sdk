# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from typing import Optional
from pydantic import BaseModel
import time
import base64
import time

from io import BytesIO

from model import QAICStableDiffusion

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run before the application starts
    print("Application startup")

    app.model = QAICStableDiffusion(device_id=0)

    yield
    # Code to run when the application shuts down
    print("Application shutdown")

app = FastAPI(lifespan=lifespan)

class ImageRequest(BaseModel):
    model: str
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = '512x512'
    response_format: Optional[str] = 'b64_json'

@app.get("/v1/models")
async def get_models():
    try:
        response = {
            "object": "list",
            "data": [
                {
                "id": "stable-diffusion-xl",
                "object": "model",
                "created": 1746296172,
                "owned_by": "system"
                }
            ],
        }

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/images/generations2")
async def generate_images2(request: Request):
    data = await request.json()
    print(data)
    return {"data": data}

@app.post("/v1/images/generations")
async def generate_images(image_request: ImageRequest):
    print(image_request)
    utc_seconds = time.time()

    size = [int(dim) for dim in image_request.size.split('x')]

    try:
        async for image in app.model.generate(image_request.prompt,
                                              image_request.n,
                                              size):
            buffered = BytesIO()
            image.save(buffered, format='PNG')
            b64_json = base64.b64encode(buffered.getvalue()).decode()

            response = {
                "created": int(utc_seconds),
                "data": [
                    {
                        "b64_json": b64_json
                    }
                ]
            }

            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

