# Roberta Model
---

## Source of the model
This models are adopted from: [Roberta-Large Trained](https://huggingface.co/deepset/roberta-large-squad2).
        
## Description of the model
---
RoBERTa is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. RoBERTa builds on BERT’s language masking strategy and modifies key hyperparameters in BERT, including removing BERT’s next-sentence pretraining objective, and training with much larger mini-batches and learning rates. RoBERTa was also trained on an order of magnitude more data than BERT, for a longer amount of time. This allows RoBERTa representations to generalize even better to downstream tasks compared to BERT.

## Framework and version
---

This model is in Pytorch and also exported to ONNX
>
        1. Pytorch CPU Version : 1.8.0
        2. Python Version: 3.6.13
        3. Transformers: 4.10.3
        4. Onnxruntime==1.9.0
        5. Onnx==1.10.1


## Execution command
---
1. Open roberta_run.sh and Choose the config you are looking for. "latency" or "throughput"
config="latency"

2. Execute the Script

## Calibration Data
---

Calibration data is needed during Quantization with INT8 precision. The Calibration data is provided using the flag: '-input-list-file=list.txt'. The list.txt shared contains only one randomly generated input. In order to attain best accuracy, list.txt should contain a wide range of examples covering the whole spectrum of inputs the model is expected to see.




