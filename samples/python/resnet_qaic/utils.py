import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

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
