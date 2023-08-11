import torch
import numpy as np
from argparse import ArgumentParser
import onnx 

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=True)
model.eval()

filename = "./inputFiles/dog.jpg"

def printResults(probabilities):
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

def runModel(args):
    # sample execution (requires torchvision)
    print("Welcome to ResNext101 Model Details")
    modelName = args.filename
    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    print(input_batch.shape)

    if args.save_input_raw:
        saveInput = input_batch.numpy().astype("float32")
        saveInput.tofile("./inputFiles/input"+ modelName +".raw")
    
    # save the ONNX and Pytorch trace Script
    if args.save_onnx:
        try:
            print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
            torch.onnx.export(model, input_batch, './generatedModels/'+ modelName +'.onnx', opset_version=11)
            # Checks
            onnx_model = onnx.load("./generatedModels/"+modelName+".onnx")  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            print('ONNX export success, saved as ' + modelName + '.onnx')
        except Exception as e:
            print('ONNX export failure: %s' % e)

    if args.save_torch_script:
        try:
            print('\nStarting TorchScript export with torch %s...' % torch.__version__)
            with torch.no_grad():
                with torch.jit.optimized_execution(False):
                    traced = torch.jit.trace(model, (input_batch))
                    traced.save("./generatedModels/"+ modelName +"Traced.pt")
                    print("Torch Trace file generated")
        except Exception as e:
            print('TorchScript export failure: %s' % e)
    
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities.shape)

    if args.run_standalone_inference:
        # Read the categories
        print("================= Pytorch StandAlone Inference FP32 ======================\n")
        printResults(probabilities)
    
    if args.run_aic_inference:
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        print("\n======================= AIC Inference FP32 =============================\n")
        output_predictions_AIC = np.fromfile("./AICOutputs/FP32/971_out_0_0.raw", dtype="float32")
        output_predictions_AIC = torch.from_numpy(output_predictions_AIC)
        probabilities = torch.nn.functional.softmax(output_predictions_AIC, dim=0)
        printResults(probabilities)

def parse_args():
    parser = ArgumentParser(description="ResNext101 Model details")
    parser.add_argument("--save-onnx",
                        action='store_true',
                        dest='save_onnx',
                        default=False,
                        help="Save the onnx Graph from Pytorch model")
    parser.add_argument("--save-torch-script",
                        action='store_true',
                        dest='save_torch_script',
                        default=False,
                        help="Save the torch script file from Pytorch model")
    parser.add_argument("--save-input-raw",
                        action='store_true',
                        dest='save_input_raw',
                        default=False,
                        help="Save the input raw files for AIC inference")
    parser.add_argument("--run-standalone-inference",
                        action='store_true',
                        dest='run_standalone_inference',
                        default=False,
                        help="Run the Standalone Pytorch Inference")
    parser.add_argument("--run-aic-inference",
                        action='store_true',
                        dest='run_aic_inference',
                        default=False,
                        help="Run the Standalone Pytorch Inference")
    parser.add_argument("--filename",
                        required=True,
                        type=str,
                        help="FileName to use for saving ONNX and TorchScript Models")
    return parser.parse_args()

def main():
    args = parse_args()
    runModel(args)

if __name__=='__main__':
    main()

    
