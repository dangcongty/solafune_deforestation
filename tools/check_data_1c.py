import numpy as np

val_paths = open('dataset/data_split256_overlap128/logging_val.txt', 'r').readlines()
train_paths = open('dataset/data_split256_overlap128/logging_train.txt', 'r').readlines()

for path in train_paths + val_paths:
    img_path = path.strip()
    mask_path = img_path.replace('train_images', 'train_masks')

    mask = np.load(mask_path)
    mask = mask[:, :, 2]
    if mask.sum() == 0:
        print