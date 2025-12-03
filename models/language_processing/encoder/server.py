# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import Optional, List,  Union
from pydantic import BaseModel
import argparse

from model import QAicEmbeddingModel

@asynccontextmanager
async def lifespan(app: FastAPI):   
    # Code to run before the application starts
    print("Application startup")

    app.model = QAicEmbeddingModel(model_name=args.model_name, qpc_path=args.qpc_path, device=args.device)

    yield

    # Code to run when the application shuts down
    print("Application shutdown")

app = FastAPI(lifespan=lifespan)

@app.get("/v1/models")
async def get_models():
    #print('get_models')
    try:
        response = {
            "object": "list",
            "data": [
                {
                "id": app.model.name,
                "object": "model",
                "created": 1746296172,
                "owned_by": "system",
                "max_model_len": 4096
                }
            ],
        }

        return response
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

class EmbeddingsRequest(BaseModel):
    model: Optional[str] = "bge-large-en-v1.5"
    input: Union[str, List[str]]
    encoding_format: Optional[str] = 'float'
    user: Optional[str] = None

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingsRequest):
    try:
        response = {'object': 'list', 'data': []}

        inputs = request.input
        if isinstance(inputs, str):
            inputs = [inputs]

        for idx, input in enumerate(inputs):
            token_embedding, sentence_embeddings = app.model.generate(input)

            response['data'].append(
                {
                    'object': 'embedding',
                    'embedding': sentence_embeddings.reshape(-1).tolist(),
                    'index': idx
                }
            )
        #print(response)
        return response
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Embedding model endpoint")

    parser.add_argument(
        "--host",
        type=str,
        help="IP address",
        default="0.0.0.0"
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Port",
        default=8000
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face auth token",
        default=None
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Hugging Face model path",
        default='BAAI/bge-large-en-v1.5'
    )

    parser.add_argument(
        "--qpc_path",
        type=str,
        help="QPC model binary path",
        default='./models/BAAI/bge-large-en-v1.5/compiled-bin-fp16-B1-C4-A3-OLS2-MOS1-best-throughput'
    )

    parser.add_argument(
        "--device",
        type=int,
        help="Cloud AI accelerator device ID",
        default=0
    )

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


