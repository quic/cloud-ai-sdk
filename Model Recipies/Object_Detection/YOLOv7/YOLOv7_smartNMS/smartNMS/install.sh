git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
git checkout 2fdc7f14395f6532ad05fb3e6970150a6a83d290
mkdir ../weights
wget  https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt   ## yolov7 weights
mv yolov7.pt ../weights/.
git apply -v ../smartNMS/yolo_export.patch 
