#Install Dependencies 
git clone https://github.com/huggingface/transformers.git
cd transformers/
git checkout 85d69a7dd1c29f9b9bca7b5a9e6b1319caf07c6b
git apply ../transformer_changes.patch
sudo python setup.py install
cd ..

pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
sudo pip install -r requirements.txt


#Generate the inputs for openai/clip-vit-base-patch16
sudo python generate_inputs.py --num_class 2 --batch_size 4
