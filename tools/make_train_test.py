import os
from glob import glob

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

mask_paths = glob('dataset/data_split256_overlap128/train_masks_split256_overlap128/*')
img_paths = [f'dataset/data_split256_overlap128/train_images_split256_overlap128/{os.path.basename(path)}' for path in mask_paths if 'json' not in path]

def get_class_distribution(mask_path, num_classes=5):
    mask = np.load(mask_path) 
    mask = mask.argmax(-1)
    total_pixels = mask.size
    class_dist = [(mask == i).sum() / total_pixels for i in range(num_classes)]
    return class_dist

class_distributions = np.array([get_class_distribution(mask) for mask in mask_paths if 'json' not in mask])

num_clusters = min(10, len(img_paths) // 5)  # Choose reasonable cluster count
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(class_distributions)

train_idx, test_idx = train_test_split(np.arange(len(img_paths)), test_size=0.2, stratify=cluster_labels)
train_images = [f'{img_paths[i]}\n' for i in train_idx]
train_masks = [mask_paths[i] for i in train_idx]
test_images = [f'{img_paths[i]}\n' for i in test_idx]
test_masks = [mask_paths[i] for i in test_idx]


with open('dataset/data_split256_overlap128/train_split256_overlap128.txt', 'w') as f:
    f.writelines(train_images)
with open('dataset/data_split256_overlap128/val_split256_overlap128.txt', 'w') as f:
    f.writelines(test_images)
# check
train_dist = np.array([get_class_distribution(mask) for mask in train_masks  if 'json' not in mask])
test_dist = np.array([get_class_distribution(mask) for mask in test_masks  if 'json' not in mask])
for i in range(5):
    print(f'Train: {train_dist[:, i].mean():.5f} Test {test_dist[:, i].mean():.5f}')