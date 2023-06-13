**Add badges to license, getting started sections** 
[Getting Started](https://docs.qualcomm.com/bundle/resource/topics/HD-PT790-991A)
# Qualcomm cloud-ai

Qualcomm Cloud AI inference accelerator cards and accompanying SDKs offer superior power and performance capabilities to meet the growing inference needs of Data Centers, Edge, and other machine learning (ML) applications. The Cloud AI Apps and Platform SDKs provide the ability to compile, optimize, quantize and run deep learning models from popular frameworks such as ONNX, PyTorch, TensorFlow, etc.

This repository contains sample applications, model recipes, tutorials, etc to aid beginner/intermediate/advanced ML application developers. This repository is a public mirror, pull requests will not be accepted. 

- For business inquiries, please contact xyz@qualcomm.com 

# Pre-requisites 
For developers with Cloud AI hardware equipped servers, refer to [Getting Started](https://docs.qualcomm.com/bundle/resource/topics/HD-PT790-991A) for   
- Overview of Cloud AI Hardware and Software Architecture 
- Installation Guide for Platform and Apps SDK 
- Release Notes of Platform and Apps SDK
- Docker support 

# Directory structure
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
        └── python-api-example-Beginner
        └── quickstart-Beginner
            └── bert-base
├── README.md

```

# Table of contents  
1. [Introduction](#introduction)  
2. [System support and SDK installation](#sdk)  
3. [Quick Start](#quick-start)  
4. [Prebuilt Samples](#prebuilt-samples)
5. [FAQ](#faq)
5. [License](#license)
6. [Resources](#resources)

Note: This repository is a public mirror, pull requests will not be accepted. Please file an issue if you have a feature or bug request.

Add points: #FIXME
1. SDK version
2. Card health, resources, and logs

# References 
- [Cloud AI Home](https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence) 
- [Cloud AI Documentation](https://docs.qualcomm.com/bundle/resource/topics/AIC_Developer_Guide)
- [Cloud AI Software Download](https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence/cloud-ai-100#Software)

