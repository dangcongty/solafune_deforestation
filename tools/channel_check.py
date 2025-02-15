

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile

image = tifffile.imread('dataset/train_images/train_0.tif')
image = np.nan_to_num(image)

pad = np.zeros((1024, 50))
imgs = np.zeros((1024, 50))
for c in range(image.shape[2]):
    imgs = np.concatenate([imgs, image[:, :, c], pad], 1)

plt.imsave('test.png', imgs)
print()

