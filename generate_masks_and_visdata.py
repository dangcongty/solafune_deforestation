import json
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from tqdm import tqdm

root = 'dataset'
os.makedirs(f'{root}/train_masks', exist_ok=True)
os.makedirs(f'{root}/vis', exist_ok=True)

with open(f'{root}/train_annotations.json', 'r') as f:
    annotations = json.load(f)
class_names = ["grassland_shrubland", "logging", "mining", "plantation"]
colors = {
    'grassland_shrubland': (255, 0, 0),
    'logging': (0, 255, 0),
    'mining': (0, 0, 255),
    'plantation': (255, 255, 255),
}
pixel_count = {
    'background': 0,
    'grassland_shrubland': 0,
    'logging': 0,
    'mining': 0,
    'plantation': 0,
}
for anno in tqdm(annotations['images']):
    file_name = anno['file_name']
    labels = anno['annotations']
    tif_image = tifffile.imread(f'dataset/train_images/{file_name}')
    h, w, c = tif_image.shape

    rgb_image = tif_image[:, :, [1, 2, 3]]
    rgb_image = np.nan_to_num(rgb_image, nan=0)
    rgb_image = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mask = np.zeros((h, w, 5)).astype(np.uint8)
    bg_mask = np.zeros((h, w)).astype(np.uint8)
    for lb in labels:
        c = lb['class']
        c_id = class_names.index(c) + 1
        poly = np.array(lb['segmentation'], dtype = float).astype(np.int32)
        mask[:, :, c_id] = cv2.fillPoly(mask[:, :, c_id].copy(), [poly.reshape((-1, 1, 2))], (1))
        pixel_count[c] += mask[:, :, c_id].sum()
        bg_mask = cv2.fillPoly(bg_mask, [poly.reshape((-1, 1, 2))], (c_id+1))
        rgb_image = cv2.polylines(rgb_image, [poly.reshape((-1, 1, 2))], True, colors[c], 2)
    bg_mask = np.where(bg_mask!=0, 0, 1)
    pixel_count['background'] += bg_mask.sum()
    # cv2.imwrite(f'{root}/vis/{file_name[:-4]}.jpg', rgb_image)
    # np.save(f'{root}/train_masks/{file_name[:-4]}.npy', mask)

total_pixels = sum(pixel_count.values())
pixel_per_class = []
for c in pixel_count:
    pixel_per_class.append(pixel_count[c]/total_pixels)
pixel_per_class = np.array(pixel_per_class)
inverse = 1.0 / pixel_per_class
inverse[0] = 0.1 # small coef learning background
# weight_classes = inverse / np.sum(inverse)
np.save('dataset/weight_classes.npy', inverse)


# data visualize
categories = list(pixel_count.keys())
values = list(pixel_count.values())
vis_colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2']  # Different colors for each category

# Plotting
plt.figure(figsize=(8, 5))
bars = plt.bar(categories, values, color=vis_colors, edgecolor='black')

# Adding titles and labels
plt.title('Pixel Count by Category', fontsize=14)
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Pixel Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar, (cat, val) in zip(bars, pixel_count.items()):
    bar.set_label(f"{cat}: {val}")
    bar.set
plt.legend(title="Categories Count", loc="upper left", fontsize=10, bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig(f'{root}/visualize_barchart.jpg')


plt.clf()
# Calculate the percentage for each category (in case values are non-zero)
total = sum(values)
if total > 0:
    percentages = [v / total * 100 for v in values]
else:
    percentages = [0 for _ in values]  # Handling the case where all values are zero

# Plotting pie chart
plt.figure(figsize=(7, 7))
plt.pie(percentages, labels=categories, autopct='%1.1f%%', startangle=90, colors=['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2'])

# Adding title
plt.title('Pixel Count by Category (Percentage)', fontsize=14)
plt.savefig(f'{root}/visualize_percentage.jpg')
