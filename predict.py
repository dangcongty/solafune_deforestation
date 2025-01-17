import json
import os
import subprocess
from glob import glob

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
import yaml
from rasterio import features
from shapely.geometry import Polygon, shape
from skimage import measure
from tqdm import tqdm

from models.unet_v2 import Unet
from tools.metrics import F1_Metrics


class Inference:
    def __init__(self, config_file = 'config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.__init_model()
        self.__init_param()

    def __init_model(self):
        model_config = self.config['Model']
        self.model = Unet(in_chan=model_config['in_channels'], num_classes=model_config['num_classes'])
        self.model.load_state_dict(torch.load(self.config['Test']['model_weight'], map_location='cpu', weights_only=True))
        self.model.eval()

    def __init_param(self):
        self.mean = np.load('dataset/mean.npy')
        self.std = np.load('dataset/std.npy')

    def normalize(self, image):
        image = np.transpose(image, (2, 0, 1)) # h w c => c h w
        image = (image - self.mean)/self.std 
        return image.astype(np.float32)
    
    def load_image(self, image_path):
        image = tifffile.imread(image_path)
        image = np.nan_to_num(image)
        image = self.normalize(image)
        return image

    def visualize_result(self):
        device = self.config['Test']['device']
        self.model.to(device)
        for path in tqdm(glob('dataset/train_images/*')):
            image = self.load_image(path)
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = self.model(image).sigmoid().cpu().numpy()
            np.save(f'outputs/train_set/{os.path.basename(path)[:-4]}.npy', pred)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)]
        for path in tqdm(glob(f'outputs/train_set/*.npy')):
            name = os.path.basename(path)[:-4]
            image = tifffile.imread(f'dataset/train_images/{name}.tif')
            rgb_image = image[:, :, [1, 2, 3]]
            rgb_image = np.nan_to_num(rgb_image, nan=0)
            rgb_image = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            label_mask = cv2.imread(f'dataset/vis/{name}.jpg')

            pred_mask = np.load(path)[0]
            pred_mask = pred_mask.argmax(0)
            pred_vis = rgb_image.copy()
            for value, color in enumerate(colors):
                pred_vis[pred_mask == value] = color
            pred_vis = rgb_image.copy()*0.5 + pred_vis*0.5

            combine = np.concatenate([label_mask*0.8, pred_vis], axis=1)
            cv2.imwrite(f'outputs/visualize/{name}.jpg', combine)

    def load_anno(self):
        class_names = ["grassland_shrubland", "logging", "mining", "plantation", "background"]
        with open("dataset/train_annotations.json", "r") as f:
            raw_annotations = json.load(f)

        truth_polygons: dict[str, dict[str, list[Polygon]]] = {}  # file_name -> class_name -> polygons
        for fn in tqdm([f"train_{i}.tif" for i in range(176)]):
            ann: dict[str, list[Polygon]] = {}  # class_name -> polygons
            for class_name in class_names:
                ann[class_name] = []

            for tmp_img in raw_annotations["images"]:
                if tmp_img["file_name"] == fn:
                    for tmp_ann in tmp_img["annotations"]:
                        poly = tmp_ann["segmentation"]
                        # convert [x1, y1, x2, y2, ..., xn, yn] to [(x1, y1), (x2, y2), ..., (xn, yn)]
                        new_poly = Polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]).buffer(0)
                        ann[tmp_ann["class"]].append(new_poly)

            truth_polygons[fn] = ann
        return truth_polygons

    def post_process(self, path):
        test_config = self.config['Test']
        class_names = ["grassland_shrubland", "logging", "mining", "plantation", "background"]
        polygons_all_imgs = {}
        for path in tqdm(glob(f'{path}/*.npy')):
            polygons_all_classes = {}
            pred_mask = np.load(path)[0]
            for i, class_name in enumerate(class_names):
                label = measure.label(pred_mask[i] > test_config['threshold'], connectivity=2, background=0).astype(np.uint8)
                polygons = []
                for p, value in features.shapes(label, label):
                    p = shape(p).buffer(0.5)
                    if p.area >= test_config['min_area']:
                        p = p.simplify(tolerance=0.5)
                        polygons.append(p)
                polygons_all_classes[class_name] = polygons
            polygons_all_imgs[os.path.basename(path).replace(".npy", ".tif")] = polygons_all_classes
        return polygons_all_imgs
    
    def get_metrics(self):
        class_names = ["grassland_shrubland", "logging", "mining", "plantation", "background"]
        val_pred_polygons = self.post_process(path = 'outputs/train_set')
        truth_polygons = self.load_anno()
        metric = F1_Metrics()
        val_f1_scores = {}

        for idx in range(176):
            fn = f"train_{idx}.tif"
            val_f1_scores[fn] = {}
            for class_name in class_names:
                pred_polys = val_pred_polygons[fn][class_name]
                truth_polys = truth_polygons[fn][class_name]
                f1_score, _, _ = metric.compute_f1(pred_polys, truth_polys)
                val_f1_scores[fn][class_name] = f1_score

        val_f1_df = pd.DataFrame(val_f1_scores).T

        val_f1_avg = val_f1_df.mean().mean()  # average of all classes and all images
        print(f"average f1 score: {val_f1_avg}")
        print("f1 scores:")

    def submission(self):
        class_names = ["grassland_shrubland", "logging", "mining", "plantation", "background"]
        device = self.config['Test']['device']
        self.model.to(device)
        for path in tqdm(glob('dataset/evaluation_images/*')):
            image = self.load_image(path)
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = self.model(image)[0].cpu().numpy()
            np.save(f'outputs/submit_set/{os.path.basename(path)[:-4]}.npy', pred)

        test_pred_polygons = self.post_process(path = 'outputs/submit_set')
        submission_save_path = "submission.json"
        images = []
        for img_id in range(118):  # evaluation_0.tif to evaluation_117.tif
            annotations = []
            for class_name in class_names:
                for poly in test_pred_polygons[f"evaluation_{img_id}.tif"][class_name]:
                    seg: list[float] = []  # [x0, y0, x1, y1, ..., xN, yN]
                    for xy in poly.exterior.coords:
                        seg.extend(xy)

                    annotations.append({"class": class_name, "segmentation": seg})

            images.append({"file_name": f"evaluation_{img_id}.tif", "annotations": annotations})

        with open(submission_save_path, "w", encoding="utf-8") as f:
            json.dump({"images": images}, f, indent=4)

if __name__ == '__main__':
    inference = Inference(config_file='config.yaml')
    inference.submission()