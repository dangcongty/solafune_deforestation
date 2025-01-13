import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
import torch.utils
import yaml
from progress_table import ProgressTable
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import Unet
from tools.dataset import Data
from tools.metrics import compute_iou


class Trainer:
    def __init__(self, config_file = 'config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.__init_losses()
        self.__init_model()
        self.__init_loader()
        self.__init_optimizer()
        self.__init_metrics()
        self.class_names = ["grassland_shrubland", "logging", "mining", "plantation", 'background']

    def __init_losses(self):
        loss_config = self.config['Losses']
        self.loss_fns = []
        weight_cls = torch.from_numpy(np.load('dataset/weight_classes.npy')).to(self.config['Train']['device'])
        if 'bce' in loss_config['type']:
            bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss(weight=weight_cls.reshape((-1, 1, 1)), smooth_factor=0.0)
            self.loss_fns.append(bce_loss_fn)
        if 'dice' in loss_config['type']:
            dice_loss_fn = smp.losses.DiceLoss(mode=smp.losses.MULTILABEL_MODE, from_logits=True)
            self.loss_fns.append(dice_loss_fn)

    def __init_model(self):
        model_config = self.config['Model']
        self.model = Unet(in_chan=model_config['in_channels'], num_classes=model_config['num_classes'])

    def __init_loader(self):
        loader_config = self.config['Loader']
        train_data = Data(mode = 'train')
        val_data = Data(mode = 'val')

        self.train_loader = DataLoader(train_data, 
                                  batch_size=loader_config['train_bs'],
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=2)
        self.val_loader = DataLoader(val_data, 
                                  batch_size=loader_config['val_bs'],
                                  shuffle=False,
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=2)
    def __init_optimizer(self):
        optim_config = self.config['Optimizer']
        optimzer = getattr(optim, optim_config["type"], None)
        self.optimizer = optimzer(self.model.parameters(), lr = optim_config['lr'])

    def __init_metrics(self):
        self.metrics = compute_iou

    def get_losses(self, output, masks):
        losses = {}
        for loss in self.loss_fns:
            name = loss.__class__.__name__
            losses[name] = loss(output, masks)
        return losses


    def train(self):
        train_config = self.config['Train']
        self.model = self.model.to(train_config['device'])
        ptable = ProgressTable(pbar_embedded=False, pbar_style="angled alt red blue")
        best_iou = 0
        for epoch in range(1, train_config['epochs']+1):
            # train
            self.model.train()
            train_losses = {
                'bce': [],
                'dice': []
            }
            ptable.update('Epoch', epoch, color="red")
            for i, (images, masks) in enumerate(ptable(self.train_loader, total=len(self.train_loader), description="Training phase")):
                images = images.to(train_config['device'])
                masks = masks.to(train_config['device'])

                outputs = self.model(images)
                losses = self.get_losses(outputs, masks)
                total_loss = train_config['bce_gain']*losses['SoftBCEWithLogitsLoss'] + train_config['dice_gain']*losses['DiceLoss']
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # log
                train_losses['bce'].append(losses['SoftBCEWithLogitsLoss'].item())
                train_losses['dice'].append(losses['DiceLoss'].item())
                ptable.update('Loss BCE', np.mean(train_losses['bce']), aggregate="mean", color="blue")
                ptable.update('Loss Dice', np.mean(train_losses['dice']), aggregate="mean", color="green")

            # evaluation
            self.model.eval()
            val_losses = {
                'bce': [],
                'dice': []
            }
            val_ious = {name: [] for name in self.class_names}
            for i, (images, masks) in enumerate(ptable(self.val_loader, total=len(self.val_loader), description="Evaluation phase")):
                images = images.to(train_config['device'])
                masks = masks.to(train_config['device'])
                with torch.no_grad():
                    outputs = self.model(images)
                losses = self.get_losses(outputs, masks)

                # metrics
                iou_all = self.metrics(outputs, masks)
                # log
                val_losses['bce'].append(losses['SoftBCEWithLogitsLoss'].item())
                val_losses['dice'].append(losses['DiceLoss'].item())
                ptable.update('Val Loss BCE', np.mean(val_losses['bce']), aggregate="mean", color="yellow")
                ptable.update('Val Loss Dice', np.mean(val_losses['dice']), aggregate="mean", color="cyan")
                for c, iou in enumerate(iou_all):
                    val_ious[self.class_names[c]].append(iou.item())
                    ptable.update(self.class_names[c], np.mean(val_ious[self.class_names[c]]), aggregate="mean", color="magenta")
                
                iou_all_classes = np.mean([np.mean(v) for k, v in val_ious.items()])
                ptable.update("All classes", iou_all_classes, aggregate="mean", color="blue")
            
            # save best model
            if iou_all_classes > best_iou:
                best_iou = iou_all_classes
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epochs': epoch
                }, f'ckpt/checkpoint.pth')
                torch.save(self.model.state_dict(), f'ckpt/best.pth')

            ptable.next_row(split=1)

        ptable.close()
    
if __name__ == '__main__':
    trainer = Trainer('config.yaml')
    trainer.train()