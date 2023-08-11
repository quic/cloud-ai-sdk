pip install -r requirements.txt
model_path=./model_dumps/
echo "model will be saved at ${model_path}"
python -m transformers.convert_graph_to_onnx --model albert-base-v2 --pipeline question-answering --framework pt ./model_dumps/albertQA.onnx

