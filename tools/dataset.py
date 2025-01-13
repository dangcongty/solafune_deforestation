import os
from glob import glob

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Data(Dataset):
    def __init__(self, mode = 'train'):
        if mode == 'train':
            self.paths = open('dataset/train.txt', 'r').readlines()
        elif mode == 'val':
            self.paths = open('dataset/val.txt', 'r').readlines()

        self.mean = np.load('dataset/mean.npy')
        self.std = np.load('dataset/std.npy')
    
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
        image_path = self.paths[index]
        image = self.load_image(image_path)
        mask_path = f'dataset/train_masks/{os.path.basename(image_path)[:-4]}npy'
        mask = np.load(mask_path).transpose((2, 0, 1))
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        return image, mask

if __name__ == '__main__':
    data = Data()
    data.__getitem__(0)