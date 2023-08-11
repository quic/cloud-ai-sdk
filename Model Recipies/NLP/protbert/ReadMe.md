# ProtBert Model
---

## Source of the model
This models are adopted from:
1. [ProtBert](https://huggingface.co/Rostlab/prot_bert).

## Description of the model
---
ProtBert is based on Bert model which pretrained on a large corpus of protein sequences in a self-supervised fashion. This means it was pretrained on the raw protein sequences only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those protein sequences.


## Framework and version
---

This model is in Pytorch and also exported to ONNX
>
        1. Pytorch CPU Version : 1.13.1
        2. Python Version: 3.8.12
        3. Transformers: 4.26.1
        4. Onnxruntime==1.14.1
        5. Onnx==1.13.1
        6. optimum==1.7.1

## Model Generation Steps (if any)
---

1. Create a "pyenv" with python 3.8.12 (pyenv install 3.8.12)
2. Activate the env. and install the requirements.txt (pip install -r requirements.txt)
3. We will generate the Onnx model using the Transformers "Optimum" library.
Run the below command in the terminal:
>
        mkdir model_files
        optimum-cli export onnx --model Rostlab/prot_bert generatedModels/ProtBert.onnx --cache_dir model_files --opset 11
4. Run the preparator post the model creation. Create a sample config (Check details on config creation in Preparator docs (attached in the gerrit too)), Run the below commands.
>
        source /opt/qti-aic/dev/python/qaic-env/bin/activate
        python modify_mul.py # This is required to change the large value of Mul for output correctness.
        python /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config prot_bert.yaml

5. This will generate the prepared model in the workspace directory.

## Generate sample inputs
The sample input is randomly generated using the following commands
>
        python generateSampleInputs.py --output-path inputFiles

## Compile & Execution commands
---
Commands to run ONNX model using qaic-exec on AIC H/W:

* Command for ONNX FP16 Version:
>
        # fp16 compile
        /opt/qti-aic/exec/qaic-exec \
          -m=ProtBert/model_modified_preparator_aic100.onnx \
          -onnx-define-symbol=batch,1 \
          -onnx-define-symbol=sequence,128 \
          -aic-hw -aic-hw-version=2.0 \
          -convert-to-fp16 \
          -aic-binary-dir=binaries-fp16
        
        # fp16 execution
        /opt/qti-aic/exec/qaic-runner \
          -t binaries-fp16 \
          -i inputFiles/input_ids.raw \
          -i inputFiles/attention_masked.raw \
          -n 1000 \
          --write-output-dir=./outputs-fp16
        
        
* Command for ONNX INT8 Version:
>
        # int8 compile
        /opt/qti-aic/exec/qaic-exec \
          -m=ProtBert/model_modified_preparator_aic100.onnx \
          -onnx-define-symbol=batch,1 \
          -onnx-define-symbol=sequence,128 \
          -aic-hw -aic-hw-version=2.0 \
          -convert-to-quantize \
          -aic-binary-dir=binaries-int8
          -execute-nodes-in-fp16=Softmax,Add,Div,Erf,Mul,LayerNorm \
        
        
        # int8 running
        /opt/qti-aic/exec/qaic-runner \
          -t binaries-int8 \
          -i inputFiles/input_ids.raw \
          -i inputFiles/attention_masked.raw \
          -n 1000 \
          --write-output-dir=./outputs-int8



## Functional Support status
---

| Platform | Status |
| ----------- | ----------- |
| AIC100 | Yes | --> ONNX Yes|
| Qognition | N/A |
| Makena | N/A |


## List of operators in this ONNX Model (Model Preparator Summary)

model_modified Summary ───────────────────────────────────────────────╮
│ IR Version: 6                                                                                                        │
│ Opset Version: 11 ,                                                                                                  │
│ Producer Name:                                                                                                       │
│ Doc:                                                                                                                 │
│ Total Count of Ops: 3061                                                                                             │
│ QModel Tool Version: 0.0.1                                                                                           │
│ Total Model Parameters: 418,921,536                                                                                  │
│                                                                                                                      │
│ All Ops: {'Unsqueeze': 243, 'Cast': 1, 'Constant': 698, 'Sub': 62, 'Mul': 122, 'Shape': 241, 'Gather': 244, 'Slice': │
│ 1, 'Add': 424, 'ReduceMean': 122, 'Pow': 61, 'Sqrt': 61, 'Div': 121, 'MatMul': 240, 'Concat': 120, 'Reshape': 120,   │
│ 'Transpose': 120, 'Softmax': 30, 'Erf': 30}                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                 model_modified Detail                                                  
╭───────────────────────────┬─────────────────────────────────────────────────────────┬───────────────────┬────────────╮
│ Name                      │ Shape                                                   │ Input/Output      │ Dtype      │
├───────────────────────────┼─────────────────────────────────────────────────────────┼───────────────────┼────────────┤
│ input_ids                 │ ['batch_size', 'sequence_length']                       │ input             │ int64      │
│ attention_mask            │ ['batch_size', 'sequence_length']                       │ input             │ int64      │
│ token_type_ids            │ ['batch_size', 'sequence_length']                       │ input             │ int64      │
│ last_hidden_state         │ ['batch_size', 'sequence_length', 1024]                 │ output            │ float32    │
╰───────────────────────────┴─────────────────────────────────────────────────────────┴───────────────────┴────────────╯
                                    Table Generated by QAicOnnxModel Preparator Tool                                    
2023-03-16 14:36:00.312 | INFO     | qaic_pytools.qmodel.preparator.preparator:summarize:150 - Final Summary of the Model
╭────────────────────────────────────── model_modified_preparator_aic100 Summary ──────────────────────────────────────╮
│ IR Version: 6                                                                                                        │
│ Opset Version: 11 , 3 ai.onnx.ml, 1 ai.onnx.training, 17 com.ms.internal.nhwc, 1 ai.onnx.preview.training, 1         │
│ com.microsoft, 1 com.microsoft.experimental, 1 com.microsoft.nchwc, 1 org.pytorch.aten, 11 com.qti.aisw.onnx,        │
│ Producer Name:                                                                                                       │
│ Doc:                                                                                                                 │
│ Total Count of Ops: 1488                                                                                             │
│ QModel Tool Version: 0.0.1                                                                                           │
│ Total Model Parameters: 418,921,995.0                                                                                │
│                                                                                                                      │
│ All Ops: {'Shape': 1, 'Gather': 4, 'Add': 394, 'Unsqueeze': 1, 'Slice': 1, 'ReduceMean': 122, 'Sub': 61, 'Pow': 61,  │
│ 'Sqrt': 61, 'Div': 121, 'Mul': 121, 'MatMul': 240, 'Reshape': 120, 'Transpose': 120, 'MaskedSoftmax': 30, 'Erf': 30} │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                        model_modified_preparator_aic100 Detail                                         
╭───────────────────────────┬─────────────────────────────────────────────────────────┬───────────────────┬────────────╮
│ Name                      │ Shape                                                   │ Input/Output      │ Dtype      │
├───────────────────────────┼─────────────────────────────────────────────────────────┼───────────────────┼────────────┤
│ input_ids                 │ ['batch_size', 'sequence_length']                       │ input             │ int64      │
│ attention_mask            │ ['batch_size', 1]                                       │ input             │ int64      │
│ token_type_ids            │ ['batch_size', 'sequence_length']                       │ input             │ int64      │
│ last_hidden_state         │ ['batch_size', 'sequence_length', 1024]                 │ output            │ float32    │
╰───────────────────────────┴─────────────────────────────────────────────────────────┴───────────────────┴────────────╯
                                    Table Generated by QAicOnnxModel Preparator Tool                                    
╭───────────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────╮
│ Stages                                                                                        │ Status               │
├───────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
│ Native Checker                                                                                │ Passed               │
│ Internal(Ort) Checker                                                                         │ Passed               │
│ Shape Inference                                                                               │ Passed               │
│ Model Simplifier Optimization                                                                 │ Passed               │
│ Pattern Optimizer Required / Not Required                                                     │ Required             │
│ Dynamic model generation                                                                      │ Skipped              │
│ Validation Checker                                                                            │ Skipped              │
╰───────────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────╯
                                             Model Preparator Stage Status

## References
---

* [ProtBert](https://huggingface.co/Rostlab/prot_bert)

## Point of contact
---

* Qranium Model Architecture Analysis Team: qranium_maa@qti.qualcomm.com

## Legal
---

* [OSR Ticket](N/A)