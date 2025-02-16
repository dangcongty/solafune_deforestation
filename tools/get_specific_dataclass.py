import os
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split

class_names = ['background', "grassland_shrubland", "logging", "mining", "plantation"]



mask_paths = glob('dataset/data_split256_overlap128/train_masks_split256_overlap128/*')
img_paths = [f'dataset/data_split256_overlap128/train_images_split256_overlap128/{os.path.basename(path)}' for path in mask_paths if 'json' not in path]

for query_class in class_names:
    data_paths = []
    query_class = 'logging'
    if query_class == 'background':
        continue
    for mask_path in mask_paths:
        mask = np.load(mask_path)
        mask = mask[:, :, class_names.index(query_class)]
        if mask.sum() == 0:
            continue
        else:
            data_paths.append(f'dataset/data_split256_overlap128/train_images_split256_overlap128/{os.path.basename(mask_path)}')

    train_idx, test_idx = train_test_split(np.arange(len(data_paths)), test_size=0.2)
    train_images = [f'{data_paths[i]}\n' for i in train_idx]
    # train_masks = [mask_paths[i] for i in train_idx]
    test_images = [f'{data_paths[i]}\n' for i in test_idx]
    # test_masks = [mask_paths[i] for i in test_idx]

    with open(f'dataset/data_split256_overlap128/{query_class}_train.txt', 'w') as f:
        f.writelines(train_images)
    with open(f'dataset/data_split256_overlap128/{query_class}_val.txt', 'w') as f:
        f.writelines(test_images)