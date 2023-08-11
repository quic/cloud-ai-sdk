#Install requirements
pip install -r requirements.txt

#Clone yolov3
git clone -b archive https://github.com/ultralytics/yolov3
cd yolov3
git apply --reject --whitespace=fix ../yoloV4WithoutModels.patch

#Generate complete graph with leaky relu
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

python3 detect.py --img-size 416 --cfg cfg/yolov4-relu.cfg --weights ./yolov4.weights

cd ..
