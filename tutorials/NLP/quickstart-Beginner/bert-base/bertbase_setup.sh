pip install -r requirements.txt

if [ -d generatedModels ]
then
  rm -rf generatedModels
fi

optimum-cli export onnx --model bert-base-cased --cache_dir model_files/cased --opset 11 --task question-answering generatedModels/ONNX/cased
