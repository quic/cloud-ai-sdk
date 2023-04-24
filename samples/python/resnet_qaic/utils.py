import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

def generate_bin(onnx_path, batch_size=1, aic_num_cores=1, precision='fp16'):
    filename, extension = os.path.splitext(onnx_path)
    onnx_folder = os.path.dirname(onnx_path)
    qpc_bin = onnx_folder+filename+'_qpc'
    if os.path.isdir(qpc_bin):
        cmd = f'sudo rm -fr {qpc_bin}'
        os.system(cmd)
        print(f'Removing existing QPC')

    cmd = f'/opt/qti-aic/exec/qaic-exec -m={onnx_path} -aic-hw -aic-hw-version=2.0 -convert-to-{precision} -onnx-define-symbol=batch_size,{batch_size} -aic-num-cores={aic_num_cores}  -aic-binary-dir={qpc_bin}'
    os.system(cmd)
    print(f'Running : {cmd}')

    return qpc_bin



# Define the data transforms
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, image_df, data_dir, transform=None):
        self.image_df = image_df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, index):
        image_path = self.data_dir+'/'+self.image_df['filename'][index]
        label = self.image_df['label'][index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label
