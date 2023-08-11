pip install -r requirements.txt
mkdir model_files
optimum-cli export onnx --model twmkn9/distilroberta-base-squad2 --cache_dir model_files --opset 13 --task question-answering distillrobertaqa
