import sys
sys.path.append("/opt/qti-aic/examples/apps/qaic-python-sdk")
import qaic
import numpy as np
import torchvision
import torch
import pandas as pd
from utils import generate_bin, CustomImageDataset, data_transforms

onnx_filename = 'resnet50.onnx'

# Create an instance of the EfficientNet-B0 model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2) 

# Export the PyTorch model to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model,                # PyTorch model
                  dummy_input,           # Input tensor
                  onnx_filename,          # Output file
                  export_params=True,   # Export the model parameters
                  opset_version=11,     # ONNX opset version
                  do_constant_folding=True,  # Fold constant values for optimization
                  input_names=['input'],    # Input tensor names
                  output_names=['output'],  # Output tensor names
                  dynamic_axes={'input': {0: 'batch_size'},  # Dynamic axes
                                'output': {0: 'batch_size'}})

# define the batch_size
batch_size = 2
# Generate binary for QAIC by default the binary is compiled for 1 nsp core, set-size = 10 and fp16 precision.
qpcPath = generate_bin(onnx_path = onnx_filename ,batch_size=batch_size, aic_num_cores=4) # return path to the folder containing compiled binary. 
#FIXME: read yaml to generate binary?

# Compile and load the model
resnet_sess = qaic.Session(model_path= qpcPath+'/programqpc.bin', options_path='./resnet_config.yaml')
input_shape, input_type = resnet_sess.model_input_shape_dict['input']
output_shape, output_type = resnet_sess.model_output_shape_dict['output']

image_df = pd.read_csv('dataset.csv') #FIXME:add Qualcomm compliant dataset.
# Define the custom dataset
dataset = CustomImageDataset(image_df, data_dir='./data', transform=data_transforms)
# Create a data loader for the dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

for batch_idx, (data, labels) in enumerate(dataloader):
    input_data = data.numpy().astype(input_type)
    input_dict = {'input': input_data}
    output = resnet_sess.run(input_dict)
    output_data = np.frombuffer(output['output'], dtype=output_type).reshape(batch_size, -1) # dtype to be modified based on given model
    predicted_labels = np.argmax(output_data, axis=1)
    print(f'Actual : {labels} vs predicted {predicted_labels}')