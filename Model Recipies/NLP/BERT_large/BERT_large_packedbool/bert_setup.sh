
sudo pip install -r requirements.txt

#Download modelfiles from mlcommons
mkdir ./mlCommonsBertFiles
wget -nc https://zenodo.org/record/3733868/files/model.ckpt-5474.data-00000-of-00001 -P ./mlCommonsBertFiles
wget -nc https://zenodo.org/record/3733868/files/model.ckpt-5474.index -P ./mlCommonsBertFiles
wget -nc https://zenodo.org/record/3733868/files/model.ckpt-5474.meta -P ./mlCommonsBertFiles
wget -nc https://zenodo.org/record/3733868/files/vocab.txt -P ./mlCommonsBertFiles
wget -nc https://raw.githubusercontent.com/mlcommons/inference/master/language/bert/bert_config.json -P ./mlCommonsBertFiles
