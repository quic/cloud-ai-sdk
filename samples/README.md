# Contents

#FIXME samples

This folder will contain examples with Python and C++ using qaicrt API. The examples will cover.

1. Float32 to Float16 conversion: The model is not optimized for AIC100, this code only shows how to get the model inference to work on AIC100. (Basic)

2. Float16 model integration: For users with fp16 model who seek to swiftly integrate it into their inference workflow, a variety of examples with pre and post processing code for CV and NLP models are provided. This aim here is to enable users to rapidly enable inference on AIC100. (Basic)

3. Model compilation for optimal performance: A walkthrough example with step-by-step instructions is provided to demonstrate how models can be compiled for AIC100 to achieve optimal throughput, and latency. (Advanced) [LINK](https://docs.qualcomm.com/bundle/resource/topics/HD-PT790-992A/End-to-End-Inference-Workflow.html)
Example on how to "Determine if host-side or device-side is the performance limiter" section 2.1.3 [LINK](https://docs.qualcomm.com/bundle/resource/topics/HD-PT790-992A/End-to-End-Inference-Workflow.html), 2.1.3.1 PCIe bandwidth usage

4. Int8 Quantization and mixed precision examples. (Advanced) [LINK 2.1](https://docs.qualcomm.com/bundle/resource/topics/HD-PT790-992A/End-to-End-Inference-Workflow.html)

5. Python and C++ APIs with best pratices: Examples to illustrate API usage and best pratices. 

6. Deployment ready code: End to End examples are provided to demonstrate running models in a production level environment. 

7. Guidance on Batch size, Set size based on input data rate. 

8. Configuration files with parameters to compile some known models in the field of NLP and CV for highest throughput, lowest latency and balanced scenarios. (own folder parallel to samples)

9. How to run more than one model on the card? Example? [oversubscription] The oversubscription feature allows scheduling of multiple networks to run on a group of NSPs in a time-shared fashion. The switching time between networks is minimized to improve overall performance (throughput and latency) using Data Plane Switching (DPS). Oversubscription is available as a precompiled tool in the Platform SDK called qaic-program-group-app. Oversubscription APIs are available for application developers.

10. custom ops example.

11. Latency measurement qaic-api-test latency stats

12. 2.1.3.3 NUMA considerations

13. 8 Network specialization example.

# Error Handling

# Benchmarking
