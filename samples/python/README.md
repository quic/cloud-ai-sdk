# Contents

This folder contains examples showing an end-to-end workflow for running inference on QAIC100 using our python APIs. 

Most of the examples follow this pattern:

1.	Get the model from open source. (HuggingFace for example)
2.	Convert the model to onnx using onnx library. (Since support for compiling onnx file is preferred)  
3.	Call generate_bin function converts onnx to qpc (#FIXME currently this takes raw arguments but I plan to swap it to use .yaml file which is currently used by qaic.session call) [TODO]
a.	Currently it is compiled for default arguments, can be replaced with best performance compile arguments) [TODO] #FIXME
4.	Creating qaic.session with appropriate input and output names.
5.	Provide sample prepossessing steps. Build input_dict for the session. 
6.	Call session.run() 
7.	Provide sample postprocessing steps. reshape output from the session. 

## Completed:
1. resnet example
2. vit example
3. MaskedLM (under llm directory) - refer to this [README](llm/README.md)

## Pending:
[ ] #FIXME add api details from https://confluence.qualcomm.com/confluence/display/LRT/Python+HL+API+Documentation 

1.	These are some new examples we need to work on:
Emphasis : since AWS interest lies in LLMs and generative AI models (Choose one from   amazon 20 list, please refer to this list for seq_len, latency and throughput requirements.)
i.	One transformer decoder model (gpt, opt) end to end model inference. 
Currently, HF models onboarded by Anuj are only compile ready. I couldn’t find an end-to-end example. 
Models with decoder potentially need some patches. 

ii.	Transformer based encoder-decoder models (one of flan, t5, bart)
Currently, HF models onboarded by Anuj are only compile ready. I couldn’t find an end-to-end example. 
Models with decoder potentially need some patches? 

iii.	Transformer based encoder only model, added bert_large_uncased as mentioned above but it is not performant for given input in fp16 precision. Works well with fp32.
1.	Can be replaced with quick-start-bert-large QA model? Need to replace inference with qaic.session. Thoughts?

iv.	Stable diffusion model


2.	Stuff we can reuse based on existing notebooks. 
We have these examples in qranium/qaic-notebooks which can be added with some modifications required for latest sdk.
a.	bert-large-quantization
b.	custom-op-demo
c.	docker
d.	Multi_Network_and_Oversubscription
e.	Performance_Tuning (covers model configurator)
f.	device_setup_health_check 
g.	qaic-notebooks/segmentation at master · qranium/qaic-notebooks (qualcomm.com)
h.	Yolo models

