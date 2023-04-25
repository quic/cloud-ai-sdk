# Steps

1.	Get the model from open source. (HF for example)
2.	Convert the model to onnx using onnx library. (Since support for compiling onnx file is preferred)  
3.	Call generate_bin function converts onnx to qpc (currently this takes raw arguments but I plan to swap it to use .yaml file which is currently used by qaic.session call) [TODO]
a.	Currently it is compiled for default arguments, can be replaced with best performance compile arguments) [TODO]
4.	Creating qaic.session with appropriate input and output names.
5.	Provide sample prepossessing steps. Build input_dict for the session. 
6.	Call session.run() 
7.	Provide sample postprocessing steps. reshape output from the session. 

