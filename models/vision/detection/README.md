## Description
---

Download the yolov5, and yolov7 models, prepare for the Qualcomm AIC100, compile for high-thoughput, min-latency, or balanced throughput with fp16 precision, run the model on a generated random sample, and obtain the benchmarking results and output values.

## Source of the models
---
The models are downloaded from (https://github.com/ultralytics/yolov5). This script has been tested for the following requested models:
* yolov5s
* yolov5m
* yolov5l
* yolov5x
* yolov7-e6e


## Virtual environment
---
For a quick environment setup:

```commandline
source /opt/qti-aic/dev/python/qaic-env/bin/activate
python3.8 -m venv det_env
source det_env/bin/activate

```

## Framework and version
---
```commandline
pip install torch==1.13.0 onnx==1.12.0 onnxruntime==1.15.0 torchvision==0.14.0 transformers==4.29.2 pandas==2.0.2 urllib3==1.26.6
pip install ultralytics seaborn nvidia-pyindex onnx-graphsurgeon

```
## Syntax
---
Copy the run_yolo_model.py and the lut_yolo_models.csv to a working directory. Pick a MODEl_NAME from the list above, and type:

```commandline

usage: run_yolo_model.py [-h] --model-name {yolov5s,yolov5m,yolov5l,yolov5x,yolov7-e6e}
             [--objective {best-latency,best-throughput,balanced}] 
	     [--opset OPSET] 
	     [--batch-size BATCH_SIZE]
             [--image-size IMAGE_SIZE] 
	     [--cores {1,2,3,4,5,6,7,8,9,10,11,12,13,14}]
             [--instances {1,2,3,4,5,6,7,8,9,10,11,12,13,14}] 
	     [--ols {1,2,3,4,5,6,7,8,9,10,11,12,13,14}] 
	     [--mos {1,2,3,4,5,6,7,8,9,10,11,12,13,14}]
             [--set-size {1,2,3,4,5,6,7,8,9,10}] 
	     [--extra EXTRA] 
	     [--time TIME] 
	     [--device {0,1,2,3,4,5,6,7}] 
	     [--run-only]



Download, Compile, and Run YOLO models on randomly generated inputs


optional arguments:
  -h, --help            show this help message and exit
  --model-name, -m {yolov5s,yolov5m,yolov5l,yolov5x,yolov7-e6e}
                        Model name to download.
  --objective, -o {best-latency,best-throughput,balanced}
                        Running for best-latency, best-throughput, or balanced
  --opset OPSET         ONNX opset. Default <12>
  --batch-size, -b BATCH_SIZE
                        Sample input batch size. Default <1>.
  --image-size, -s IMAGE_SIZE
                        Sample input image width/height. Default <640>.
  --cores, -c {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Number of AIC100 cores to compile the model for. Default <2>
  --instances, -i {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Number of model instances to run on AIC100. Default <7>
  --ols {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Overlap split factor. Default <1>
  --mos {1,2,3,4,5,6,7,8,9,10,11,12,13,14}             
                        Maximum output channel split. Default <None>
  --set-size {1,2,3,4,5,6,7,8,9,10}
                        Set size. Default <10>
  --extra EXTRA         Extra compilation arguments.
  --time TIME           Duration (in seconds) for which to submit inferences. Default <20>
  --device, -d {0,1,2,3,4,5,6,7}
                        AIC100 device ID. Default <0>
  --run-only, -r        Performs the inference only, without re-exporting and re-compiling the model


```
Examples:
```commandline
python run_yolo_model.py -m yolov5s -o best-throughput
```
```commandline
python run_yolo_model.py -m yolov5m -o balanced
```
```commandline
python run_yolo_model.py -m yolov5x -o best-throughput
```

The hardware configuration will be either associated to the corresponding row in the lut_yolo_models.csv or to defualt values if not specified by the user. If the MODEL_NAME is not included in the lut_yolo_models.csv, default values will be used.

After download, compile, and run is complete, the working directory of the selected model looks as follows. 
# Working directory structure
```
|── model                   # Contains the onnx file of the picked model 
|   └── model.onnx          # The onnx file of the picked model
|── inputFiles              # Contains the (randomly generated) input files of the compiled model
│   └── input_img*.raw      # Randomly generated input files of the compiled model
|── outputFiles             # Contains the corresponding output to input, as well as the hardware profiling for latency
│   └── fp16*               
│       └── output-act*.bin # Corresponding output to the randomly generated input_img*.raw
│       └── aic-profil*.bin # The hardware profiling for round trip latency between host and device for each inference
├── compiled-bin*           # Compilation path
│   └── programqpc.bin      # For the selected objective, the model.onnx is compiled into programqpc.bin 
├── list*.txt               # A list that contains path to the inputs. Can be used as input to qaic-runner
├── commands*.txt           # Includes necessary compilation and running scripts to reproduce the results manually.

```
To manually resproduce the results, navigate to the working directory, select the qaic compile/run commands from the command*.txt and run them in the terminal. 

