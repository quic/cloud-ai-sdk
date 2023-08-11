#Install Requirements
pip install -r requirements.txt

mkdir generatedModels

#Save model as ONNX
sudo python Model.py --filename ResNet101 --save-onnx --save-torch-script

