pip install numpy==1.23.4
pip install onnx==1.8.0 onnxruntime==1.8.0
pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout c016dbdbdaf79339ae6d275d4651dc9f380be055
cd ..
pip install -e transformers
pip install onnx-simplifier==1.8.0
pip install sympy
pip install protobuf==3.19.0
