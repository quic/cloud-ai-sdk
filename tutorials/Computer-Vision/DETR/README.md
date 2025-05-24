## Description
---

Download the DETR-ResNet50 model, prepare for the Qualcomm AIC100, compile the model, run the model on a generated random sample along with input image, and obtain the output.


## Source of the model
---

This model is an implementation of DETR-ResNet50 found at (https://github.com/facebookresearch/detr).


## Virtual environment
---
For a quick environment setup:

```commandline
python3.8 -m venv cv_workflow_env
source cv_workflow_env/bin/activate
pip install --upgrade pip

```

## Framework and version
---
```commandline
pip install torch==2.4.1+cpu torchvision==0.19.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.24.4 onnx==1.17.0 pillow==10.4.0 requests==2.32.3 notebook==7.3.3 matplotlib==3.7.5 scipy==1.10.1

```

