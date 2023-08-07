Tutorials are Jupyter notebools designed to walk the developer through the Cloud AI inference workflow. The tutorials are split into 2 categories - CV and NLP. Overall, the inference workflow for the CV and NLP are quite similar. The differences are primarily related to the usage of the tools in Cloud AI SDK.   

`Model-Onboarding` - This is one of the beginner notebooks. This goes through exporting and preparing the model, compiling the model using a CLI tool and executing inference using CLI tool / Python APIs. 

`Performance-Tuning` - This is another beginner notebook that walks the developer through the key parameters to optimize for best performance (latency and throughput) on Cloud AI platforms. Going through this notebook and 'Performance Tuning' section in the Quick start guide will equip developers with a intuitive understanding of how to use the key parameters to meet inference application KPIs (AI compute resource usage, throughput and latency).   

`Profiler` - This is a intermediate-level notebook that describes system and device level inference profiling capabilities. Developers can use the tools and techniques described in this tutorial to measure application/device level latency and identify system/device bottlenecks. 

`Accuracy-Evaluator` - This is a intermediate-level notebook that describes usage of the Accuracy Evaluator tool. Accuracy evaluator is used to evaluate the accuracy of the Cloud AI inference outputs vs a reference implementation like ONNX Runtime for an input dataset. Developers have the ability to choose the accuracy metrics to be reported by the tool. 

`Accuracy-Analyzer` - This is a intermediate-notebook that walks the developer through the Accuracy analyzer tool used in debugging accuracy issues on a model compiled for Cloud AI. The tool provides multiple techniques/strategies to identify the operations causing the accuracy issues. 

`Quantization` - This is an intermediate-level notebook that goes over post training quantization techniques supported by Cloud AI SDK. This builds on the accruacy evaluator and analyzer notebooks. 

### Pre-requisites
Install qaic python package <br>
`/opt/qti-aic/dev/python/qaic-env/bin/python3.8 install /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl`

### Jupyter Notebook Setup 

`/opt/qti-aic/dev/python/qaic-env/bin/pip install ipykernel`

`/opt/qti-aic/dev/python/qaic-env/bin/python -m ipykernel install --user --name qaic-env --display-name "qaic-env"`

`source /opt/qti-aic/dev/python/qaic-env/bin/activate`

Clone the Cloud AI repo and run `jupyter notebook --allow-root --ip 0.0.0.0 --no-browser`



