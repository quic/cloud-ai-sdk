# This folder consists

1. `vit_qaic` and `resnet_qaic` folder contains example showing an end-to-end workflow for running inference on QAIC100 using the python APIs. 
2. `qaic_features` folder consists of examples to show how to perform benchmarking, profiling and measuring metrics for inferences made on the device.

# Installation

Steps to install `qaic` API:

```
pip install /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
pip install -r requirements.txt
```


## Structure of end to end workflow

Examples follow this pattern:

1.	Get the model from open source. (HuggingFace for example)
2.	Convert the model to onnx using onnx library. 
3.	Call generate_bin function converts onnx to qpc (binary for the device).
a.	Currently it is compiled for default arguments, can be replaced with best performance compile arguments) #FIXME
4.	Creating `qaic.session` with appropriate input and output names.
5.	Provide sample prepossessing steps. Build input_dict for the session. 
6.	Call session.run() to perform inference.
7.	Provide sample postprocessing steps. reshape output from the session. 

## To run the example

```
python example.py 
```
