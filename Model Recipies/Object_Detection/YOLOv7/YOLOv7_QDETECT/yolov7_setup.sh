bash QDetect/install.sh 
pip3 install -r yolov7/requirements.txt
pip3 install onnx
cd yolov7
cp ../QDetect/create_model.py .
mkdir ONNX
python create_model.py --name yolov7 --h_w 640 640
cp -rf ONNX/ ../.
cd ..
