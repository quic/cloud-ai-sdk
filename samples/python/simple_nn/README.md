# Description
This example demonstrates how to run a compiled simple neural network inference on Qualcomm AI100 using Python API.
The simple_nn.onnx model is compiled with qaic-exec command shown below:
/opt/qti-aic/exec/qaic-exec -m=./simple_nn.onnx -aic-hw -aic-hw-version=2.0 -convert-to-fp16 -batchsize=1 -aic-num-cores=1 -compile-only -aic-binary-dir=./simple_nn_bin

# Prerequisites
TODO:

# Running the sample
python infer.py
TODO: Description of threaded version. 

# Links
TODO: place links to relevant portions of document here.

# Python APIs used

```http
  inferenceSet
```  

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `id` | `class` | **Required**. blah |


