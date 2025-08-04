# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from transformers import AutoTokenizer
import numpy as np
import qaic

class QAicEmbeddingModel():
    def __init__(self, model_name='BAAI/bge-large-en-v1.5', qpc_path='./models/BAAI/bge-large-en-v1.5/compiled-bin-fp16-B1-C4-A3-OLS2-MOS1-best-throughput', device=0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.aic_session = qaic.Session(model_path=os.path.join(qpc_path, 'programqpc.bin'), dev_id=device)
        self.name = model_name

        self.aic_session.setup()

    def generate(self, input):
        tokens = self.tokenizer(input, padding=True, return_tensors='np')

        input_data = {'input_ids': None,
                      'attention_mask': None}

        for k in input_data.keys():
            input_shape, input_type = self.aic_session.model_input_shape_dict[k]

            rows, cols = tokens[k].shape
            input_data[k] = np.zeros(input_shape, dtype=input_type)
            input_data[k][:rows, :cols] = tokens[k]

        outputs = self.aic_session.run(input_data)

        output_shape, output_type = self.aic_session.model_output_shape_dict['token_embeddings']
        token_embeddings = np.frombuffer(outputs['token_embeddings'], dtype=output_type).reshape(output_shape)
        token_embeddings = token_embeddings[:, 0]

        output_shape, output_type = self.aic_session.model_output_shape_dict['sentence_embedding']
        sentence_embedding = np.frombuffer(outputs['sentence_embedding'], dtype=output_type).reshape(output_shape)

        return token_embeddings, sentence_embedding

def main():
    inputs_txt = 'your_text_here'
    model = QAicEmbeddingModel()
    token_embedding, sentence_embeddings = model.generate(inputs_txt)
    print('token_embedding {}'.format(token_embedding))
    print('sentence_embeddings {}'.format(sentence_embeddings))

if __name__ == "__main__":
    main()
