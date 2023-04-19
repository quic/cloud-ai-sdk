import sys
sys.path.append("/opt/qti-aic/examples/apps/qaic-python-sdk")
import qaic
import numpy as np
import torchvision
import torch
import pandas as pd
from utils import generate_bin, CustomImageDataset, data_transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import onnx
from onnxsim import simplify


# assert int(sys.argv[1]) in [224, 384], "Input size should be 224 or 384."
image_size = 224

#onnx_filename = f"./vit-base-patch16-{int(sys.argv[1])}-in21k.onnx"
onnx_filename = "vit.onnx"

# Create an instance of the EfficientNet-B0 model
model = ViTForImageClassification.from_pretrained(f'google/vit-base-patch16-{image_size}')

# Export the PyTorch model to ONNX
dummy_input = torch.randn(1, 3, image_size, image_size).type(torch.FloatTensor)
torch.onnx.export(model,                # PyTorch model
                  dummy_input,           # Input tensor
                  onnx_filename,          # Output file
                  export_params=True,   # Export the model parameters
                  opset_version=11,     # ONNX opset version
                  do_constant_folding=True,  # Fold constant values for optimization
                  input_names=['image'],    # Input tensor names
                  output_names=['output'],  # Output tensor names
                  dynamic_axes={'image': {0: 'batch_size'},  # Dynamic axes
                                'output': {0: 'batch_size'}})

# FIXME: is onnxsim required?
onnx_model = onnx.load(onnx_filename)
onnx_model_simp, check = simplify(onnx_model)
onnx.save(onnx_model_simp, onnx_filename)
print("ONNX model saved at: ", onnx_filename)

# define the batch_size
batch_size = 1
# Generate binary for QAIC by default the binary is compiled for 1 nsp core, set-size = 10 and fp16 precision.
qpcPath = generate_bin(onnx_path = onnx_filename ,batch_size=batch_size, aic_num_cores=4) # return path to the folder containing compiled binary. 
#FIXME: read yaml to generate binary?

# Compile and load the model
resnet_sess = qaic.Session(model_path= qpcPath+'/programqpc.bin', options_path='./vit_config.yaml')
input_shape, input_type = resnet_sess.model_input_shape_dict['image']
output_shape, output_type = resnet_sess.model_output_shape_dict['output']

# image_df = pd.read_csv('dataset.csv') #FIXME:add Qualcomm compliant dataset.
# # Define the custom dataset
# dataset = CustomImageDataset(image_df, data_dir='./data', transform=data_transforms)
# # Create a data loader for the dataset
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# for batch_idx, (data, labels) in enumerate(dataloader):
#     input_data = data.numpy().astype(input_type)
#     input_dict = {'image': input_data}
#     output = resnet_sess.run(input_dict)
#     output_data = np.frombuffer(output['output'], dtype=output_type).reshape(batch_size, -1) # dtype to be modified based on given model
#     predicted_labels = np.argmax(output_data, axis=1)
#     print(f'Actual : {labels} vs predicted {predicted_labels}')

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

batch_size = 1
device = True
if device:
    input_data = inputs['pixel_values'].numpy().astype(input_type)
    input_dict = {'image': input_data}
    output = resnet_sess.run(input_dict)
    logits = np.frombuffer(output['output'], dtype=output_type).reshape(batch_size, -1) # dtype to be modified based on given model
else:
    outputs = model(**inputs)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])