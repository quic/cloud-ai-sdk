# Installation

```
pip install /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
pip install -r requirements.txt
```


# Contents

This folder contains examples showing an end-to-end workflow for running inference on QAIC100 using the python APIs. 

Most of the examples follow this pattern:

1.	Get the model from open source. (HuggingFace for example)
2.	Convert the model to onnx using onnx library. 
3.	Call generate_bin function converts onnx to qpc.
a.	Currently it is compiled for default arguments, can be replaced with best performance compile arguments) [TODO] #FIXME
4.	Creating qaic.session with appropriate input and output names.
5.	Provide sample prepossessing steps. Build input_dict for the session. 
6.	Call session.run() 
7.	Provide sample postprocessing steps. reshape output from the session. 

# Execute

To run the example:

```
python example.py 
```
