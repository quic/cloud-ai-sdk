####################################################################################################
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import os
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
import numpy as np
import torch
import qaic

model = 'openai/whisper-base'
aic_path = './whisper_AIC'
#aic_model = './binaries/whisper-tiny-c4/programqpc.bin'

# Select an audio file and read it:
ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
audio_sample = ds[0]['audio']
waveform = audio_sample['array']
sampling_rate = audio_sample['sampling_rate']

# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained(model)
model = WhisperForConditionalGeneration.from_pretrained(model)
#print(model.config)

# Use the model and processor to transcribe the audio:
input_features = processor(
    waveform, sampling_rate=sampling_rate, return_tensors='pt'
).input_features

# Generate token ids
predicted_ids = model.generate(input_features)

print('predicted_ids CPU:')
print(predicted_ids)

# Decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print('CPU response: {}'.format(transcription))

whisper_sess = qaic.Session(model_path=os.path.join(aic_path, 'programqpc.bin'))
whisper_sess.setup()

features_shape, features_type = whisper_sess.model_input_shape_dict['input_features']
decoder_shape, decoder_type = whisper_sess.model_input_shape_dict['decoder_input_ids']
output_shape, output_type = whisper_sess.model_output_shape_dict['logits']

# Set start token to 'startoftranscript|>' (50257)
decoder_input_ids = torch.tensor([[50257]])

input_dict = {
     'input_features': input_features.numpy().astype(features_type),
     'decoder_input_ids': decoder_input_ids.numpy().astype(decoder_type)
}

#print('input_features:')
#print(input_features)
#print('input_dict:')
#print(input_dict)

output = whisper_sess.run(input_dict)
#print('Model output:')
#print(output)

predicted_ids = np.frombuffer(output['logits'], dtype=output_type).reshape(output_shape) # dtype to be modified based on given model

print('predicted_ids AIC:')
print(predicted_ids[0])

transcription = processor.batch_decode(predicted_ids[0], skip_special_tokens=True)
print('AIC response: {}'.format(transcription))
