####################################################################################################
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import os
from datasets import load_dataset
from transformers import WhisperProcessor
import whisper
import numpy as np
import torch
import qaic

model_name = 'base'
aic_path = './whisper_AIC'

# Select an audio file and read it:
ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
audio_path = ds[0]['audio']['path']
audio = whisper.load_audio(audio_path) # Read audio from file
audio_pad = whisper.pad_or_trim(audio) # Padding and trimming
# make log-Mel spectrogram and move to the same device as the model
input_features = whisper.log_mel_spectrogram(audio_pad) # convert to mel spectrogram

# Load the Whisper processor for parsing results
processor = WhisperProcessor.from_pretrained('openai/whisper-{}'.format(model_name))

eot = 50257 # end of transcript token
startoftranscript = 50258 # start of transcript token

decoder_sequence_length=150

def run_AIC(input_features, device_id=0):
    # Load both encoder and decoder models into Cloud AI accelerator memory
    # via oversubscription.
    # The number of NSP cores required is the maximum of the numbers of cores
    # for which encoder and decoder are compiled.
    # If encoder is compiled for 4 cores and decoder is compiled for 12 cores,
    # then the max usage is 12 cores.
    # Since encoder and decoder don't run at the same time, this allows us to
    # efficiently utilize the available cores.

    encoder_sess = qaic.Session(
        model_path=os.path.join(aic_path, 'whisper-encoder', 'programqpc.bin'),
        num_activations=1,
        set_size=1,
        dev_id=device_id,
        oversubscription_name='group1')

    decoder_sess = qaic.Session(
        model_path=os.path.join(aic_path, 'whisper-decoder', 'programqpc.bin'),
        num_activations=1,
        set_size=1,
        dev_id=device_id,
        oversubscription_name='group1')

    encoder_inputs = {
        'input_features': input_features.numpy().astype(np.float32).reshape(1,80,3000)
    }

    audio_features = encoder_sess.run(encoder_inputs)['last_hidden_state']

    next_token = None
    tokens = [startoftranscript]
    decoder_input_ids = np.zeros((1, decoder_sequence_length), dtype=np.int64)
    decoder_input_ids[:,0] = startoftranscript

    for iter in range(decoder_sequence_length):
        if iter > 0:
            decoder_input_ids[:,iter] = next_token.item()

        decoder_inputs = {
            'input_ids': decoder_input_ids,
            'encoder_hidden_states': audio_features,
        }

        logits = decoder_sess.run(decoder_inputs)['logits']
        logits = logits[:,iter,:]

        next_token = logits.argmax(axis=-1)
        tokens.append(next_token.item())

        if next_token == eot: # stop at end-of-transcript token
            break

    transcription = processor.batch_decode(tokens, skip_special_tokens=False)
    print("result:", transcription)

if __name__ == '__main__':
    run_AIC(input_features)
