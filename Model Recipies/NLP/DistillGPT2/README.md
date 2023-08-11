# DistilGPT2 Model
---

## Source of the model
This model is adopted from Hugging Face[DistilGPT2](https://huggingface.co/distilgpt2).

## Description of the model
---

DistilGPT2 English language model pretrained with the supervision of GPT2 (the smallest version of GPT2) on OpenWebTextCorpus, a reproduction of OpenAI's WebText dataset. The model has 6 layers, 768 dimension and 12 heads, totalizing 82M parameters (compared to 124M parameters for GPT2). On average, DistilGPT2 is two times faster than GPT2.

* distilgpt2.onnx
   1. original model from hugging_face repo
* distilgpt2_onetoken.onnx
   1. removed past/present from input and output respectively
   2. added a slice before the final fc layer
* distilgpt2_onetoken_stage1.onnx
   1. removed past/present from input and output respectively
   2. added a slice before the final fc layer
   3. added new input next_token_idx
   4. argmax after final logits to return token_ids

## Framework and version
---

*  This model is exported to ONNX
   1. ONNX-Runtime CPU Version : 1.8.1
   2. Python Version: 3.8
   3. Transoformers: 4.10.3

## Execution command
---

1. Execute run_distilgpt2.sh
 

## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.


