#Install Requirements
sudo pip install -r requirements.txt

#Download Model using optimum-cli
mkdir model_files
optimum-cli export onnx --model Rostlab/prot_bert generatedModels/ --cache_dir model_files --opset 11
