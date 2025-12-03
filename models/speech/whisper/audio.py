####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import os
import numpy as np
from datasets import load_dataset, Audio
import soundfile as sf
from pathlib import Path

class AudioSample:
    def __init__(self):
        # load dummy dataset and read soundfiles
        self.ds = load_dataset(
            'hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation'
        )

    def to_file(self, parent='.'):
        audio_sample = self.ds[0]['audio']

        audio_array  = audio_sample['array']
        audio_fname = os.path.join(parent, Path(audio_sample['path']).name)
        sampling_rate = audio_sample["sampling_rate"]

        # Convert to float32 for compatibility with soundfile
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        try:
            sf.write(audio_fname, audio_array, sampling_rate, format='FLAC')
        except Exception as e:
            print('Error saving file: {}'.format(e))

        return audio_fname