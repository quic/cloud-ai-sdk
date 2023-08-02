**Add badges for license, documentation etc** 
[Getting Started](https://docs.qualcomm.com/bundle/resource/topics/HD-PT790-991A)
# Qualcomm cloud-ai

Qualcomm Cloud AI inference accelerator cards and accompanying SDKs offer superior power and performance capabilities to meet the growing inference needs of Data Centers, Edge, and other machine learning (ML) applications. The Cloud AI Apps and Platform SDKs provide the ability to compile, optimize, quantize and run deep learning models from popular frameworks such as ONNX, PyTorch, etc.

This repository contains sample applications, model recipes, tutorials, etc to aid beginner/intermediate/advanced ML application developers. This repository is a public mirror, pull requests will not be accepted. 


# Inference on Cloud AI

Cloud AI SDKs enable developers to optimize trained deep learning models for high-performance inference. The SDKs provide workflows to optimize the models for best performance,  provides runtime for execution and supports integration with ONNXRT and Triton Inference Server for deployment.

Cloud AI SDKs support 
- Generative AI, Natural Language Processing, Recommender systems and Computer Vision models running on Cloud AI hardware performantly
- Optimize performance of the models per application requirements (throughput, accuracy and latency) through various quantization techniques
- Development of inference applications through support for multiple OS and docker containers.  
- Deploy inference applications at scale with support for Triton (**trademark**) inference server

To execute inference on Cloud AI platforms, a pre-trained model, trained using ML framework such as PyTorch or TensorFlow, is exported to ONNX format. The ONNX file is compiled using the Apps SDK to generate a binary file (QPC). Developers can also use graph API in the Apps SDK to create the network graph and compile it. The binary file (QPC) is executed on the Cloud AI accelerator, using the Platform runtime, aided by the kernel drivers and the device firmware.  

# Cloud AI SDK
An Application and Platform SDK constitute the Cloud AI SDK. 

The Application (Apps) SDK consists of model development tools, including a sophisticated parallelizing compiler, performance and integration tools, and code samples. 

The Platform SDK consists of development tools icluding a kernel space runtime, which contains the API's and language bindings, accompanied by kernel drivers, a user space runtime, card firmware, and several card monitoring, telemetry, profiling and debugging tools.  

![Cloud AI SDK](images/Plat_Apps_SDK.JPG) 

# Contents of this repository  
1. API_Reference_Card - This section will cover the APIs developers can use to execute inferences.
    - C++ APIs 
    - Python APIs for inference
    - ONNX Runtime support on Cloud AI 
2. docs - This section covers the end-to-end inference workflow, from onboarding a pre-trained model to execution of inferences on Cloud AI platforms. 
    - Export and prepare the model 
    - Compile and optimize the model 
    - Execute, integrate and deploy in production pipeline
    - Platform management 
5. models - This section provides the best throughput and best latency recipes for many CV, NLP and LLMs. 
6. samples - #FIXME
7. tutorials - This section breaks down the inference workflow on Cloud AI into multiple tutorials (Jupyter notebooks) across NLP/CV that developers can peruse. 

# Pre-requisites
For developers with Cloud AI hardware equipped servers, refer to [Getting Started](https://docs.qualcomm.com/bundle/resource/topics/80-PT790-991A) for   
- Overview of Cloud AI Hardware and Software Architecture 
- Installation Guide for Platform and Apps SDK 
- Release Notes of Platform and Apps SDK

# Directory structure (FIXME)
```
.
├── API_Reference_Card      # Provides quick overview of Python and C++ APIs to interact with Cloud AI hardware
├── docker                  # dockerfiles
|── models                  # Contains model recipes for computer vision, natural language processing and generative AI models 
|── samples                 # End-to-end inference examples on Cloud AI using CPP and Python APIs
│   └── cpp                 # CPP based APIs used inference
│   └── python              # Python based APIs used for inference (#FIXME Amey add details about subdirectories)
├── tools                   # Cloud AI tools (key tools like - qaic-exec, qaic-runner, accruacy evaluator, accuracy analyzer etc) 
├── triton                  # Example using triton inference server
├── tutorials               # Quick Start Guides
    └── Computer-Vision
    └── Generative-AI
    └── NLP
        └── Accuracy-Analyzer-Intermediate
        └── Accuracy-Evaluator-Intermediate
        └── Model-Onboarding-Beginner
        └── Performance-Tuning-Beginner
        └── Quantization-Intermediate 
        └── Sample-Application-Beginner
├── README.md

```
# References (FIXME)
- [Cloud AI Home](https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence) 
- [Cloud AI Software Download](https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence/cloud-ai-100#Software)

