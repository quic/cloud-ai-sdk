#Install Requriments
pip install -r requirements.txt

#Conver model from https://huggingface.co/bert-large-uncased to ONNX
sudo python generateModel.py --model-name google/electra-base-discriminator --model-class ElectraForPreTraining 

