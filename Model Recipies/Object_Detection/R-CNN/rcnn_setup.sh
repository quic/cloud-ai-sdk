#Install Dependencies
sudo pip install -r requirements.txt

#Clone and Install Detectron2
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && git checkout 42ef395311e8f4d9ff189416ba89f7c9c3d07258 && cd ..
sudo python -m pip install -e detectron2

#For dataset, refer Coco Dataset 2017 https://cocodataset.org/#download
#Detectron2 require the dataset diretory to have a specific directory structure, refer Detectron2 Documentation at https://detectron2.readthedocs.io/tutorials/builtin_datasets.html#expected-dataset-structure-for-coco-instance-keypoint-detection

echo "Enter the location of Coco dataset folder Val2017: "
read dataset_folder
export DETECTRON2_DATASETS=$dataset_folder


#After successful installation, one can export the model to ONNX 
#For installation issues, refer https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#common-installation-issues

sudo pip install protobuf==3.19.0

cd detectron2/
git apply ../scripts/416x416_pre1000_post200_SR0.patch


cd tools/deploy

python ./export_model.py \
       	--config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	--output ./onnx_model_maskrcnn \
	--format onnx \
	MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
	MODEL.DEVICE cpu

cd ../../../
