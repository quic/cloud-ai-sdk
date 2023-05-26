# Qualcomm cloud-ai

Qualcomm Cloud AI inference accelerator cards and accompanying SDKs offer superior power and performance capabilities to meet the growing inference needs of Data Centers, Edge, and other machine learning (ML) applications. The Cloud AI Apps and Platform SDKs provide the ability to compile, optimize, quantize and run deep learning models from popular frameworks such as ONNX, PyTorch, TensorFlow, etc.

This repository contains sample applications, model recipes, tutorials, etc to aid beginner/intermediate/advanced ML application developers. This repository is a public mirror, pull requests will not be accepted. 

- For business inquiries, please contact xyz@qualcomm.com 

# Access to Cloud AI Instances 
### AWS 
The AWS Cloud AI 100 instances have the Platform and Apps SDK installed. Developers can clone this repository on the instance to get started.  

# Pre-requisites 
For developers with Cloud AI equipped servers, refer to [Getting Started](https://docs.qualcomm.com/bundle/resource/topics/HD-PT790-991A) for   
- Overview of Cloud AI Hardware and Software Architecture 
- Installation Guide for Platform and Apps SDK 
- Release Notes of Platform and Apps SDK
- Docker support 

# Directory structure
```
.
├── API_Reference_Card      # Provides quick overview of Python and C++ APIs to interact with Cloud AI hardware
├── docker                  # dockerfiles
|── images                  # Images related to documentation (Should remove this and host images elsewhere)
|── models                  # Config and script files for some popular models (#FIXME Rohan please add your files here)
|── samples                 # End-to-end inference examples on Cloud AI
│   ├── cpp                 # CPP based APIs used for QAIC100 inference
│   ├── python              # Python based APIs used for QAIC100 inference (#FIXME Amey add details about subdirectories)
└── triton                  # example using triton inference server
└── tutorials               # Jupyter notebook based tutorials
  - Gettting Started
    - README.md
    - NLP
      - bert-base
        - tutorial1
            -nb1.ipynb
            -nb1.ipynb
    - CV

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

