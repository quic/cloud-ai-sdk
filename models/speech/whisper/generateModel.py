####################################################################################################
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def main(model_name: str):
    # load model and processor
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # load dummy dataset and read soundfiles
    ds = load_dataset(
        'hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation'
    )
    input_features = processor(
        ds[0]['audio']['array'], return_tensors='pt'
    ).input_features

    # Generate logits
    logits = model(input_features, decoder_input_ids=torch.tensor([[50258]])).logits
    # take argmax and decode

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    decoder_input_ids = torch.tensor([[50258]])
    torch_outputs = model(input_features, decoder_input_ids=decoder_input_ids)
    output_names = list(torch_outputs.keys())

    pt_model_code = model.forward.__code__
    pt_input_names = pt_model_code.co_varnames[1 : pt_model_code.co_argcount]

    # Save the input files for AIC compilation.
    # input_features.numpy().tofile('inputFiles/input_features.raw')
    # decoder_input_ids.numpy().tofile('inputFiles/decoder_input_ids.raw')

    onnx_model_name = f"{model_name.split('/')[1]}.onnx"
    torch.onnx.export(
        model,
        (input_features, decoder_input_ids),
        onnx_model_name,
        input_names=pt_input_names,
        output_names=output_names,
    )

    # Run the OnnxRuntime Inference
    #inputs = {'input_features': input_features, 'decoder_input_ids': decoder_input_ids}

    #import onnxruntime

    #ort_session = onnxruntime.InferenceSession(onnx_model_name)
    #ort_outputs = ort_session.run(None, {k: v.numpy() for k, v in inputs.items()})

    #for orto, oname in zip(ort_outputs, ort_session.get_outputs()):
        #print(orto.shape, oname.name)
        #orto.flatten().tofile(f'./AICOutputs/ort_full/{oname.name}.raw')


if __name__ == '__main__':
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument(
        '--model-name',
        required=True,
        help='Model name to generate',
    )
    args = argp.parse_args()
    main(**vars(args))
