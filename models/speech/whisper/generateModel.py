####################################################################################################
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import os
import argparse
import numpy as np
import torch
from datasets import load_dataset
import whisper

def main(model_name: str, output_dir: str):
    cache_path = './cache'
    # load dummy dataset and read soundfiles
    ds = load_dataset(
        'hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation'
    )

    audio_sample = ds[0]['audio']

    audio = whisper.load_audio(audio_sample['path']) # Read audio from file
    audio_pad = whisper.pad_or_trim(audio) # Padding and trimming

    # make log-Mel spectrogram and move to the same device as the model
    input_features = whisper.log_mel_spectrogram(audio_pad) # convert to mel spectrogram
    input_features = torch.unsqueeze(input_features, 0) # add batch dimension

    model = whisper.load_model(model_name, download_root=cache_path)
    audio_features = model.encoder(input_features)
    decoder_input_ids = torch.tensor([[50258]])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Encoder model
    torch.onnx.export(
        model.encoder,
        (input_features),
        os.path.join(output_dir, 'encoder_model.onnx'),
        input_names=['input_features'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_features': {0: 'batch_size', 1: 'feature_size', 2: 'encoder_sequence_length'},
            'last_hidden_state': {0: 'batch_size'}
        }
    )

    # Decoder model
    torch.onnx.export(
        model.decoder,
        (decoder_input_ids, audio_features),
        os.path.join(output_dir, 'decoder_model.onnx'),
        input_names=['input_ids', 'encoder_hidden_states'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'decoder_sequence_length'},
            'encoder_hidden_states': {0: 'batch_size', 1: 'encoder_sequence_length'},
            'logits': {0: 'batch_size', 1: 'decoder_sequence_length'}
        }
    )

if __name__ == '__main__':
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument(
        '--model-name',
        required=True,
        help='Model name to generate',
    )
    argp.add_argument(
        '--output-dir',
        required=False,
        help='Path to store generated ONNX files',
        default='./'
    )
    args = argp.parse_args()
    main(**vars(args))
