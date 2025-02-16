import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.getcwd())
from tools import Augmentation


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
    def __init__(self, mode = 'train', config = None):
        
        self.mean = np.load('dataset/mean.npy')
        self.std = np.load('dataset/std.npy')

        if mode == 'train':
            self.paths = open('dataset/data_split256_overlap128/logging_train.txt', 'r').readlines()
        elif mode == 'val':
            self.paths = open('dataset/data_split256_overlap128/logging_val.txt', 'r').readlines()

        self.config = config
        self.down = self.config['Loader']['down']

        self.augs = Augmentation()
        self.mode = mode

    
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

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index].strip()
        mask_path = img_path.replace('train_images', 'train_masks')
        t1 = time.time()
        image = np.load(img_path)
        image = self.normalize(image)
        image = np.stack([image[3], image[4], image[7], image[10], image[11]], 0)
        t2 = time.time()
        mask = np.load(mask_path)
        mask = mask[:, :, 2] # 2: logging
        h, w = image.shape[1:]

        if self.mode == 'train':
            try:
                image, mask = self.augs.flip_lr(image, mask, 0.5)
                image, mask = self.augs.flip_ud(image, mask, 0.5)
                # image, mask = self.augs.rotate(image, mask, 90, 0.5)
                # image, mask = self.augs.translate(image, mask, 5, 0.5)
            except Exception as e:
                print(e)
        t3 = time.time()
        mask = torch.from_numpy(mask.copy())
        image = torch.from_numpy(image.copy())
        t4 = time.time()
        # resize 1024x1024
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), (1024, 1024), mode='nearest').squeeze().to(torch.long)
        image = F.interpolate(image.unsqueeze(0).float(), (1024, 1024), mode='nearest').squeeze()
        t5 = time.time()
        # print(f'image: {t2-t1:.4f} aug: {t3-t2:.4f} toTorch: {t4-t3:.4f} upscale: {t5-t4:.4f}')
        return image, mask

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    data = Data(mode='train', config = config)
    data.__getitem__(0)