import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor
from transformers import ViTImageProcessor, ViTForImageClassification
import yaml
from PIL import Image
import numpy as np

# Define the data transforms
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocessImg(img_path, batch_size):
    if not os.path.isfile(img_path):
        print("Error: Image path not found : ", img_path)
        exit()

    single_image = Image.open(img_path)

    feature_extractor = ViTFeatureExtractor.from_pretrained(\
                                            'google/vit-base-patch16-224-in21k')
    single_img_data = feature_extractor(images=single_image, \
                                                return_tensors="np")
    preprocessed_single_img = single_img_data['pixel_values']
    # preprocessed_single_img : [1 x 3 x 224 x 224] : Numpy array
    preprocessed_batch = np.tile(preprocessed_single_img, \
                                                [batch_size, 1, 1, 1])
    return preprocessed_batch

class CustomImageDataset(Dataset):
    def __init__(self, image_df, data_dir, transform=None):
        self.image_df = image_df
        self.data_dir = data_dir
        self.transform = transform
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, index):
        image_path = self.data_dir+'/'+self.image_df['filename'][index]
        label = self.image_df['label'][index]
        image = Image.open(image_path)#.convert('RGB')
        preprocessed_image = self.processor(images=image, return_tensors="pt")['pixel_values']
        preprocessed_image = preprocessed_image.view(3,224,224)
        # # image = Image.open(image_path).convert('RGB')
        # image = preprocessImg(image_path, 1)
        # # img_tensor, _ = preprocessImg(args.img_path, args.input_size)

        # # if self.transform is not None:
        #     # image = self.transform(image)

        return preprocessed_image, label
