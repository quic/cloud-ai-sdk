#Install Requriments
pip install -r requirements.txt

#Conver model from https://huggingface.co/bert-base-cased to ONNX
sudo python generateModel.py --model-name bert-base-uncased --model-class AutoModelForMaskedLM

