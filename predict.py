import json
import os
import subprocess
from glob import glob

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
import yaml
from rasterio import features
from shapely.geometry import Polygon, shape
from skimage import measure
from tqdm import tqdm

from models.unet_v2 import Unet
from models.yolo import YOLOSeg


class Inference:
    def __init__(self, config_file = 'config.yaml'):
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.__init_model()
        self.__init_param()

        self.class_names = ['background', "grassland_shrubland", "logging", "mining", "plantation"]
    def __init_model(self):
        model_config = self.config['Model']
        self.model = YOLOSeg(config_file = model_config['config_file'], scale = model_config['scale'])
        # self.model = Unet(in_chan=model_config['in_channels'], num_classes=model_config['num_classes'])
        self.model.load_state_dict(torch.load(self.config['Test']['model_weight'], map_location='cpu', weights_only=True))
        self.model.eval()
        
    def __init_param(self):
        self.mean = np.load('dataset/mean.npy')
        self.std = np.load('dataset/std.npy')


    def load_and_split(self, img_path, trainset = False):
        imgsz = self.config['Test']['imgsz']
        overlap = self.config['Test']['overlap']
        split_size = self.config['Test']['split_size']
        name = os.path.basename(img_path)[:-4]
        image = tifffile.imread(img_path)
        image = np.nan_to_num(image)
        h, w = image.shape[:2]

        if trainset:
            mask = np.load(img_path.replace('train_images', 'train_masks')[:-4]+'.npy')
            mask = mask.argmax(-1)
            masks = []

        split_images = []
        split_bgr = []
        for i in range(0, h-overlap, overlap):
            for j in range(0, w-overlap, overlap):
                split_img = image[i:i+split_size, j:j+split_size]

                if trainset:
                    split_mask = mask[i:i+split_size, j:j+split_size]
                    masks.append(split_mask)
                bgr_image = cv2.normalize(split_img.copy(), None, 0, 255, cv2.NORM_MINMAX)[:, :, 1:4].astype(np.uint8)
                
                split_img =  np.transpose(split_img, (2, 0, 1))
                split_img = (split_img - self.mean)/self.std 
                split_img = torch.from_numpy(split_img.copy())
                split_img = F.interpolate(split_img.unsqueeze(0).float(), (imgsz, imgsz), mode='nearest')
                split_images.append(split_img)
                split_bgr.append(bgr_image)
                
        if trainset:
            return split_images, split_bgr, masks  
        else:
            return split_images, split_bgr, split_bgr
    
    def postprocess(self, mask, min_area = 5000):
        polygons_all_classes = {}
        for i, class_name in enumerate(self.class_names):
            mask_class = (mask==i)*1
            label = measure.label(mask_class, connectivity=1, background=0).astype(np.uint8)
            polygons = []
            for p, value in features.shapes(label, label):
                p = shape(p).buffer(0.5)
                if p.area >= min_area:
                    p = p.simplify(tolerance=0.5)
                    polygons.append(p)
            polygons_all_classes[class_name] = polygons
        return polygons_all_classes

    def predict(self, data_path = 'dataset/evaluation_images'):
        device = self.config['Test']['device']
        threshold = self.config['Test']['threshold']
        overlap = self.config['Test']['overlap']
        split_size = self.config['Test']['split_size']
        self.model.to(device)
        polygons = {}
        images = []
        submission_save_path = "submission.json"
        for img_path in tqdm(glob(f'{data_path}/*')):
            annotations = []
            conf_maps = []
            pred_maps = []
            split_images, split_bgr, masks = self.load_and_split(img_path)
            with torch.no_grad():
                for split_img, split_vis, mask in zip(split_images, split_bgr, masks):
                    split_img = split_img.to(device)
                    output = self.model(split_img)[0][0]
                    output_conf = output.softmax(0)
                    output_cls = output.argmax(0)
                    max_confidence, predicted_classes  = torch.max(output_conf, dim = 0)
                    conf_maps.append(output_conf)
                    pred_maps.append(predicted_classes)
                
            result_map = torch.zeros((5, 1024, 1024)).to(device)
            _, h, w = result_map.shape
            for ci, i in enumerate(range(0, h-overlap, overlap)):
                for cj, j in enumerate(range(0, w-overlap, overlap)):
                    cmap = conf_maps[3*ci + cj]
                    cmap = F.interpolate(cmap.unsqueeze(0).float(), (512, 512)).squeeze()
                    result_map[:, i:i+split_size, j:j+split_size] = torch.maximum(result_map[:, i:i+split_size, j:j+split_size], cmap)
            class_map = torch.argmax(result_map, 0).cpu().numpy().astype(np.uint8)
            
            polygon = self.postprocess(class_map, threshold)
            for c in self.class_names:
                for poly in polygon[c]:
                    seg = [] 
                    for xy in poly.exterior.coords:
                        seg.extend(xy)
                    annotations.append({"class": c, "segmentation": seg})

            images.append({"file_name": f"{os.path.basename(img_path)}", "annotations": annotations})


            # visualize
            vis_pred = np.stack([class_map, class_map, class_map], -1)
            vis_img = tifffile.imread(img_path)
            vis_img = vis_img[:, :, [1, 2, 3]]
            vis_img = np.nan_to_num(vis_img, nan=0)
            vis_img = cv2.normalize(vis_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # gt_path = f'dataset/vis/{os.path.basename(img_path)[:-4]}.jpg'
            # gt = cv2.imread(gt_path)
            # vis = np.concatenate([gt, np.zeros((vis.shape[0], 50, 3)), vis*50], 1)
            cv2.imwrite(f'outputs/visualize/{os.path.basename(img_path)[:-4]}.jpg', vis_pred *50+vis_img*0.8)
            
        with open(submission_save_path, "w", encoding="utf-8") as f:
            json.dump({"images": images}, f, indent=4)

if __name__ == '__main__':
    infer = Inference()
    infer.predict()