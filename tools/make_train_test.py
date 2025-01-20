from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split

data_paths = glob('dataset/train_images/*')

for i in range(5):
    trains, vals = train_test_split(data_paths, test_size=0.2, random_state=np.random.randint(0, 100000), shuffle=True)

    with open(f'dataset/kfold/train_{i}.txt', 'w') as f:
        for train in trains:
            f.write(f'{train}\n')

    with open(f'dataset/kfold/val_{i}.txt', 'w') as f:
        for val in vals:
            f.write(f'{val}\n')