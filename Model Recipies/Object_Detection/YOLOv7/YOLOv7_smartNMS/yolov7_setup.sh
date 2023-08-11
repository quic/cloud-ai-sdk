bash smartNMS/install.sh 
pwd
pip3 install -r yolov7/requirements.txt
pip3 install onnx
cd yolov7
cp ../smartNMS/create_model.py .
pwd
mkdir ONNX
python create_model.py --name yolov7 --h_w 640 640
cp -r ONNX ../.
rm -rf ONNX
cd ..
