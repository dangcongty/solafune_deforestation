import os
from glob import glob

import cv2
import numpy as np
import tifffile
from tqdm import tqdm

split_size = 256
overlap = 128
images_dir = f'dataset/train_images_split{split_size}_overlap{overlap}'
masks_dir = f'dataset/train_masks_split{split_size}_overlap{overlap}'
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)
for path in tqdm(glob('dataset/train_images/*.tif')):
    name = os.path.basename(path)[:-4]
    image = tifffile.imread(path)
    image = np.nan_to_num(image)
    
    mask = np.load(f'dataset/train_masks/{name}.npy')
    
    vis_image = cv2.normalize(image.copy(), None, 0, 255, cv2.NORM_MINMAX)[:, :, 1:4].astype(np.uint8)
    h, w = image.shape[:2]
    for i in range(0, h-overlap, overlap):
        for j in range(0, w-overlap, overlap):
            split_img = image[i:i+split_size, j:j+split_size]
            split_mask = mask[i:i+split_size, j:j+split_size]
            np.save(f'{images_dir}/{name}_{i}_{j}_{i+split_size}_{j+split_size}.npy', split_img)
            np.save(f'{masks_dir}/{name}_{i}_{j}_{i+split_size}_{j+split_size}.npy', split_mask)
            # vis_image = cv2.rectangle(vis_image, (i, j), (i+split_size, j+split_size), (np.random.random(3)*255).astype(int).tolist(), 1)
            