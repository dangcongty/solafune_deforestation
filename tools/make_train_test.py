from glob import glob

from sklearn.model_selection import train_test_split

data_paths = glob('dataset/train_images/*')

trains, vals = train_test_split(data_paths, test_size=0.2, random_state=317201, shuffle=True)


with open('dataset/train.txt', 'w') as f:
    for train in trains:
        f.write(f'{train}\n')

with open('dataset/val.txt', 'w') as f:
    for val in vals:
        f.write(f'{val}\n')