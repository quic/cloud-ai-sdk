import torch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b0')
model.set_swish(False)

shape = (1,3,224,224)

rand_input = torch.rand(shape)

rand_input.shape

output_path = 'efficientnetb0.onnx'

torch.onnx.export(model, rand_input, output_path, input_names=['inp'], output_names=['out'], opset_version=11)
