import os
from glob import glob
import random

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import torch.nn.functional as F

def set_seed(seed=3107):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed()
class Data(Dataset):
    def __init__(self, mode = 'train', datatype = 'contrastive'):
        if mode == 'train':
            self.paths = open('dataset/train.txt', 'r').readlines()
            self.images = np.load('dataset/images_train.npy')
            self.masks = np.load('dataset/masks_train.npy')
        elif mode == 'val':
            self.paths = open('dataset/val.txt', 'r').readlines()
            self.images = np.load('dataset/images_val.npy')
            self.masks = np.load('dataset/masks_val.npy')

        self.mean = np.load('dataset/mean.npy')
        self.std = np.load('dataset/std.npy')
        class_names = ["grassland_shrubland", "logging", "mining", "plantation", 'background']

        # self.images = []
        # self.masks = []
        # for path in self.paths:
        #     mask_path = f'dataset/train_masks/{os.path.basename(path)[:-4]}npy'
        #     self.images.append(self.load_image(path))
        #     self.masks.append(np.load(mask_path).transpose((2, 0, 1)))
        # self.images = np.stack(self.images)
        # self.masks = np.stack(self.masks)
        # np.save('dataset/images_val.npy', self.images)
        # np.save('dataset/masks_val.npy', self.masks)

        self.datatype = datatype

    def normalize(self, image):
        image = np.transpose(image, (2, 0, 1)) # h w c => c h w
        image = (image - self.mean)/self.std 
        return image.astype(np.float32)

    def get_meanstd(self):
        train_paths = open('dataset/train.txt', 'r').readlines()
        val_paths = open('dataset/val.txt', 'r').readlines()
        num_channels = 12
        sum_images = np.zeros(num_channels)
        sum_squared_images = np.zeros(num_channels)
        total_pixels = 0
        for path in tqdm(train_paths + val_paths):
            path = path.strip()
            image = tifffile.imread(path)
            image = np.nan_to_num(image)
            for c in range(num_channels):
                sum_images[c] += np.sum(image[..., c])
                sum_squared_images[c] += np.sum(image[..., c]**2)
            
            total_pixels += image.size // num_channels
        mean = sum_images / total_pixels
        std = np.sqrt((sum_squared_images / total_pixels) - mean**2)
        for c in range(num_channels):
            print(f"Channel {c}:")
            print(f"  Mean: {mean[c]}")
            print(f"  Standard Deviation: {std[c]}")
        mean = mean.reshape(12, 1, 1)
        std = std.reshape(12, 1, 1)
        np.save('dataset/mean.npy', mean)
        np.save('dataset/std.npy', std)

    def load_image(self, image_path):
        image_path = image_path.strip()
        image = tifffile.imread(image_path)
        image = np.nan_to_num(image)
        image = self.normalize(image)
        return image

    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        mask = mask.argmax(0) 

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if self.datatype == 'contrastive':
            index2 = np.random.randint(0, len(self.images))
            image2 = self.images[index2]
            mask2 = self.masks[index2]
            mask2 = mask2.argmax(0) 

            image2 = torch.from_numpy(image2)
            mask2 = torch.from_numpy(mask2)
            return image, mask, image2, mask2

        return image, mask

if __name__ == '__main__':
    data = Data()
    data.__getitem__(0)