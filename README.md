**Add badges for license, documentation etc** 


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

