from glob import glob
import numpy as np
import shutil
import os
from tqdm import tqdm

os.makedirs(f'dataset/filter/0', exist_ok=True)
os.makedirs(f'dataset/filter/1', exist_ok=True)
os.makedirs(f'dataset/filter/2', exist_ok=True)
os.makedirs(f'dataset/filter/3', exist_ok=True)
for mask_path in tqdm(glob(f'dataset/train_masks/*')):
    vis_path = f'dataset/vis/{os.path.basename(mask_path)[:-4]}.jpg'
    mask = np.load(mask_path).argmax(-1)
    bincount = np.bincount(mask.flatten())
    for idx, c in enumerate(bincount[1:]):
        if c > 0:
            shutil.copy(vis_path, vis_path.replace('vis', f'filter/{idx}'))