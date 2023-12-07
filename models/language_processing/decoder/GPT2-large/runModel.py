from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import transformers

from qaic_infer import QAICInferenceSession


def main(model_name: str, prompt_len: int, ctx_len: int, qpc: str, prompt: str, device_id: List[int] = [0]):

    model_name = model_name
    qpc = qpc
    prompt_len = prompt_len
    ctx_len = ctx_len
    prompt = prompt

    # Load QPC
    session = QAICInferenceSession(qpc, device_id)

    # Skip inputs/outputs
    session.skip_outputs(set([x for x in session.output_names if x.endswith("_RetainedState")]))
    session.skip_inputs(set([x for x in session.input_names if x.startswith("past_")]))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare inputs for first iteration
    start = perf_counter()
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=prompt_len)
    batch_size, prompt_len = inputs["input_ids"].shape
    inputs["position_ids"] = (np.cumsum(inputs["attention_mask"], 1) - 1) * inputs["attention_mask"]
    inputs["attention_mask"] = np.concatenate(
        [
            inputs["attention_mask"].astype(bool),
            np.zeros((batch_size, ctx_len - prompt_len), dtype=bool),
        ],
        1,
    )
    cache_index = np.array([0])
    inputs["cache_index"] = cache_index
    print(prompt, end="", flush=True)

    # Run prefill
    outputs = session.run(inputs)
    next_token_id = outputs["logits"].argmax(2)
    inputs["input_ids"] = next_token_id
    inputs["position_ids"] = inputs.pop("attention_mask").sum(1, keepdims=True)
    cache_index += prompt_len
    print(tokenizer.batch_decode(next_token_id), end="", flush=True)

    # Skip attention_mask from next iteration to use retained attention_mask
    session.skip_inputs({"attention_mask"})

    loop_start = perf_counter()
    while (next_token_id != tokenizer.eos_token_id).all() and cache_index < ctx_len:
        outputs = session.run(inputs)

        # Prepare inputs for next iteration
        next_token_id = outputs["logits"].argmax(2)
        inputs["input_ids"] = next_token_id
        inputs["position_ids"] += 1
        cache_index += 1

        print(tokenizer.batch_decode(next_token_id), end="", flush=True)

    end = perf_counter()

    print()
    print("E2E:", round((cache_index.item() - prompt_len) / (end - start), 2), "tok/s")
    print("Loop:", round((cache_index.item() - prompt_len - 1) / (end - loop_start), 2), "tok/s")


if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument("--model-name", required=True, help="Model name to run the model")
    argp.add_argument("--prompt-len", type=int, default=128, help="Prompt length")
    argp.add_argument("--ctx-len", type=int, default=512, help="Context length")
    argp.add_argument("--qpc", required=True, help="QPC")
    argp.add_argument("--prompt", default="My name is", help="Input prompt to generate for")
    argp.add_argument("--device_id", default=[0], type=lambda device_ids: [int(x) for x in device_ids.split(",")], help="QAIC device ids (comma-separated)")

    args = argp.parse_args()
    main(**vars(args))
