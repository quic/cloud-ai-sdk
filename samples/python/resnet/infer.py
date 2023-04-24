import sys
sys.path.append("/opt/qti-aic/dev/lib/x86_64/")
import numpy as np
from utils import QAIC_Inference, generate_bin, CustomImageDataset, data_transforms
from torch.utils.data import DataLoader
import pandas as pd
import torchvision
import torch

image_df = pd.read_csv('dataset.csv')
# Define the custom dataset
dataset = CustomImageDataset(image_df, data_dir='./data', transform=data_transforms)
# Create a data loader for the dataset
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Create an instance of the EfficientNet-B0 model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2) 

# Export the model to ONNX format
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'resnet50.onnx')

# Generate binary for QAIC by default the binary is compiled for 1 nsp core, set-size = 10 and fp16 precision.
qpcPath = generate_bin('resnet50.onnx',batch_size=batch_size) # return path to the folder containing compiled binary. 
qaic = QAIC_Inference(qpcPath)

for batch_idx, (data, labels) in enumerate(dataloader):
    input_data = data.numpy().astype(np.float32)
    output_arr = qaic.infer(input_data, inf_id=batch_idx)
    output_data = np.frombuffer(output_arr, dtype=np.float32).reshape(batch_size, -1) # dtype to be modified based on given model
    predicted_labels = np.argmax(output_data, axis=1)
    print(f'Actual : {labels} vs predicted {predicted_labels}')