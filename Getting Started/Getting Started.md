# Getting Started with Cloud AI 

Cloud AI SDKs enable developers to optimize trained deep learning models for high-performance inference. The SDKs provide workflows to optimize the models for best performance,  provides runtime for execution and supports integration with ONNXRT and Triton Inference Server for deployment.

Cloud AI SDKs support 
- Generative AI, Natural Language Processing, and Computer Vision models running on Cloud AI hardware performantly
- Optimize performance of the models per application requirements (throughput, accuracy and latency) through various quantization techniques
- Development of inference applications through support for multiple OS and docker containers.  
- Deploy inference applications at scale with support for [Triton](https://github.com/triton-inference-server/server) inference server

## Cloud AI SDKs
An Application and Platform SDK constitute the Cloud AI SDK. 

The Application (Apps) SDK consists of model development tools, including a sophisticated parallelizing compiler, performance and integration tools, and code samples. Apps SDK is supported on  

The Platform SDK consists of development tools icluding a kernel space runtime, which contains the API's and language bindings, accompanied by kernel drivers, a user space runtime, card firmware, and several card monitoring, telemetry, profiling and debugging tools.  

![Cloud AI SDK](../images/Plat_Apps_SDK.JPG) 

## [Installation](https://github.qualcomm.com/qranium/cloud-ai/blob/main/Getting%20Started/Installation/installation.md)
The installation guide covers 
- Platforms, operating systems and hypervisors supported and corresponding pre-requisites
- Cloud AI SDK (Platform and Apps SDK) installation
- Docker support

## [Quick Start Guide](https://github.qualcomm.com/qranium/cloud-ai/blob/main/Getting%20Started/Inference%20Workflow/Inference%20Workflow.md) 
The Quick start guide provides the inference workflow on Cloud AI SDK, from onboarding a pre-trained model to deployment on Cloud AI platforms. 

## [Release Notes](https://docs.qualcomm.com/bundle/80-PT790-1/resource/80-PT790-1.pdf)
Cloud AI release notes provide developers with new features, limitations and modifications in the Platform and Apps SDKs.   

## [Tutorials](https://github.qualcomm.com/qranium/cloud-ai/tree/main/tutorials)
Tutorials, in the form of Jupyter Notebooks walk the developer through the Cloud AI inference workflow as well as the tools used in the process. Tutorials are divided into CV and NLP to provide a better developer experience even though the inference workflows are quite similar. 

## [Model Recipes](https://github.qualcomm.com/qranium/cloud-ai/tree/main/models)
Model recipes provide the developer the most performant and efficient way to run some of the popular models across categories. The recipe starts with the public model. The model is then exported to ONNX, some patches are applied if required, compiled and executed for best performance. Developers can use the recipe to integrate the compiled binary into their inference application.   

## [Sample Code](https://github.qualcomm.com/qranium/cloud-ai/tree/main/samples)
Sample code helps developers get familair with the usage of Python and C++ APIs for inferencing on Cloud AI platforms. 

 




