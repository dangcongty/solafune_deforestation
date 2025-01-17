import numpy as np
import tifffile
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import yaml
import os
import sys
sys.path.append(os.getcwd())
from models.unet_v2 import Unet
from torch.utils.data import DataLoader
from tools.dataset import Data
import torch.nn.functional as F


mean = np.load('dataset/mean.npy')
std = np.load('dataset/std.npy')
def normalize( image):
    image = np.transpose(image, (2, 0, 1)) # h w c => c h w
    image = (image - mean)/std 
    return image.astype(np.float32)

def load_image( image_path):
    image_path = image_path.strip()
    image = tifffile.imread(image_path)
    image = np.nan_to_num(image)
    image = normalize(image)
    return image

config_file = 'config.yaml'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
model = Unet(in_chan=config['Model']['in_channels'], num_classes=config['Model']['num_classes'])
model.load_state_dict(torch.load('ckpt/v1/best.pth', weights_only=True))
model.eval()
model.to('cuda:0')

loader_config = config['Loader']
# train_data = Data(mode = 'train', datatype='contrastive')
# val_data = Data(mode = 'val', datatype='normal')

images, masks = load_image('dataset/train_images/train_1.tif'), np.load('dataset/train_masks/train_1.npy').transpose((2, 0, 1))
masks = masks.argmax(0) 
images = torch.from_numpy(images)
masks = torch.from_numpy(masks)
images = images.to('cuda:0')
masks = masks.to('cuda:0')
with torch.no_grad():
    outputs, feat = model(images.unsqueeze(0))
    # outputs_sup, feat_sup = model(images_sup)
features1 = F.interpolate(feat, size = (64, 64), mode='bilinear', align_corners=False)
targets1 = F.interpolate(masks.unsqueeze(0).float().unsqueeze(1), size = (64, 64), mode='nearest').squeeze(1)

# features2 = F.interpolate(feat_sup, size = (64, 64), mode='bilinear', align_corners=False)
# targets2 = F.interpolate(masks_sup.float().unsqueeze(1), size = (64, 64), mode='nearest').squeeze(1)

# t1_flat = targets1.view(1, -1)[0] 
# t2_flat = targets2.view(1, -1)[0] 
# o2_flat = features2.view(1, 4, -1)[0].permute((1, 0)).cpu().numpy()
                                   
features = features1.view(1, 16, -1)[0].permute((1, 0)).cpu().numpy()
targets = targets1.view(1, -1)[0].cpu().numpy().astype(int)


# Apply t-SNE to reduce to 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42)  # Default perplexity should be fine for 4096 samples
features_2d = tsne.fit_transform(features)

# Plot the results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=targets, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.savefig('tSNE.jpg')
