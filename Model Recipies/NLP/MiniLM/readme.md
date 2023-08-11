# MiniLM-Uncased-Squad2 Model
---

## Source of the model
This model is adopted from Hugging Face[MiniLM-Uncased-Squad2](https://huggingface.co/deepset/minilm-uncased-squad2/blob/main/config.json) with Question Answering Head for Squad 2.0.

The untrained model is adopted from from [MiniLM-L12-H384-uncased](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased) and we have added Q&A head over it. The head part of this model is untrained.

## Description of the model
---

This model is MiniLM model with Question Answering head trained on Squad 2.0 dataset. MiniLM has the same Transformer architecture as BERT, and is compressed using deep self-attention distillation. The small model (student) is trained by deeply mimicking the self-attention module, which plays a vital role in Transformer networks, of the large model (teacher). MiniLM (12-layer, 384-hidden) achieves 2.7x speedup and comparable results over BERT-Base (12-layer, 768-hidden) on NLU tasks as well as strong results on NLG tasks.

## Framework and version
---

This model is exported to ONNX
1. onnx==1.10.2
2. onnxruntime==1.9.0
3. onnx-simplifier==0.3.6
4. Python 3.8.12
5. torch==1.10.0+cpu
6. Transoformers: (https://github.com/huggingface/transformers.git)


## Execution Steps
---

1. Execute minilm_setup.sh

2. Open minilm_setup.sh and  Configure the following options as per your choice:

Select 'fp16' precision
precision='fp16'

Choose the config you are looking for. Select 1 For best Throughput or 2 for best Latency.
config=1

3. Execute the Script


## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.


