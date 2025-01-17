import os
import random
import shutil

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

from models.unet_v2 import Unet
from tools.dataset import Data
from tools.metrics import compute_iou
from tools.losses import DiceLoss, PixelWiseLabel, CEFocalLoss

from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=3107):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed()  
class Trainer:
    def __init__(self, config_file = 'config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.__init_losses()
        self.__init_model()
        self.__init_loader()
        self.__init_optimizer()
        self.__init_metrics()
        self.__init_workspace()
        self.class_names = ["grassland_shrubland", "logging", "mining", "plantation", 'background']

    def __init_workspace(self):
        current_ver = len(os.listdir('ckpt')) + 1
        self.save_model_dir = f'ckpt/v{current_ver}'
        os.makedirs(self.save_model_dir, exist_ok=True)
        shutil.copyfile('config.yaml', f'{self.save_model_dir}/config.yaml')
        shutil.copyfile('main.py', f'{self.save_model_dir}/main.py')
        shutil.copytree('tools', f'{self.save_model_dir}/tools')
        shutil.copytree('models', f'{self.save_model_dir}/models')

        self.writer = SummaryWriter(log_dir=self.save_model_dir+'/log') 

    def __init_losses(self):
        loss_config = self.config['Losses']
        self.loss_fns = []
        num_classes = self.config['Model']['num_classes']
        weight_cls = torch.from_numpy(np.load('dataset/weight_classes.npy')).to(self.config['Train']['device']).to(torch.float)
        if 'focal' in loss_config['type']:
            focal_loss_fn = CEFocalLoss(weights=weight_cls, num_classes=num_classes)
            self.loss_fns.append(focal_loss_fn)
        if 'dice' in loss_config['type']:
            dice_loss_fn = DiceLoss(num_classes=num_classes, reduction='mean', weight=weight_cls.reshape((-1)))
            self.loss_fns.append(dice_loss_fn)
        if 'ct' in loss_config['type']:
            pwl_loss_fn = PixelWiseLabel()
            self.loss_fns.append(pwl_loss_fn)

    def __init_model(self):
        model_config = self.config['Model']
        self.model = Unet(in_chan=model_config['in_channels'], num_classes=model_config['num_classes'])

    def __init_loader(self):
        loader_config = self.config['Loader']
        train_data = Data(mode = 'train', datatype='contrastive')
        val_data = Data(mode = 'val', datatype='normal')

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

    def get_losses(self, outputs, masks, feat, outputs_sup, masks_sup, feat_sup, contrast = True):
        losses = {}
        for loss in self.loss_fns:
            name = loss.__class__.__name__
            if name == 'CEFocalLoss':
                losses[name] = 0
                losses[name] += loss(outputs.to(torch.float), masks.long())
                if contrast:
                    losses[name] += loss(outputs_sup.to(torch.float), masks_sup.long())
            elif name == 'PixelWiseLabel':
                losses[name] = loss(feat.to(torch.float), masks.long(), feat_sup.to(torch.float), masks_sup.long(), contrast)
            else:
                losses[name] = 0
                losses[name] += loss(outputs.to(torch.float), masks.long())
                if contrast:
                    losses[name] += loss(outputs_sup.to(torch.float), masks_sup.long())
        return losses


    def train(self):
        train_config = self.config['Train']
        self.model = self.model.to(train_config['device'])
        ptable = ProgressTable(pbar_embedded=False, pbar_style="angled alt red blue")
        best_iou = 0
        train_ious = {name: [] for name in self.class_names}
        for epoch in range(1, train_config['epochs']+1):
            # train
            self.model.train()
            train_losses = {
                'bce': [],
                'dice': [],
                'ct': []
            }
            ptable.update('Epoch', epoch, color="red")
            for i, ((images, masks, images_sup, masks_sup)) in enumerate(ptable(self.train_loader, total=len(self.train_loader), description="Training phase")):
                images = images.to(train_config['device'])
                masks = masks.to(train_config['device'])

                images_sup = images_sup.to(train_config['device'])
                masks_sup = masks_sup.to(train_config['device'])

                outputs, feat = self.model(images)
                outputs_sup, feat_sup = self.model(images_sup)
                losses = self.get_losses(outputs, masks, feat, outputs_sup, masks_sup, feat_sup)

                total_loss = train_config['focal_gain']*losses['CEFocalLoss'] \
                            + train_config['dice_gain']*losses['DiceLoss'] \
                            + train_config['ct_gain']*losses['PixelWiseLabel']
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # log
                train_losses['bce'].append((train_config['focal_gain']*losses['CEFocalLoss']).item())
                train_losses['dice'].append((train_config['dice_gain']*losses['DiceLoss']).item())
                train_losses['ct'].append((train_config['ct_gain']*losses['PixelWiseLabel']).item())

                ptable.update('Loss BCE', np.mean(train_losses['bce']), aggregate="mean", color="blue")
                ptable.update('Loss Dice', np.mean(train_losses['dice']), aggregate="mean", color="green")
                ptable.update('Loss CT', np.mean(train_losses['ct']), aggregate="mean", color="red")

                iou_all = self.metrics(outputs, masks)
                for c, iou in enumerate(iou_all):
                    train_ious[self.class_names[c]].append(iou.item())                
                iou_all_classes = np.mean([np.mean(v) for k, v in train_ious.items()])
                ptable.update("Train IoU", iou_all_classes, aggregate="mean", color="blue")

            self.writer.add_scalar("Train_Loss/BCE", np.mean(train_losses['bce']), epoch)
            self.writer.add_scalar("Train_Loss/Dice", np.mean(train_losses['dice']), epoch)
            self.writer.add_scalar("Train_Loss/PixelWiseLabel", np.mean(train_losses['ct']), epoch)

            for c, iou in enumerate(iou_all):
                train_ious[self.class_names[c]].append(iou.item())
                self.writer.add_scalar(f"Train_IoU/{c}", np.mean(train_ious[self.class_names[c]]), epoch)

            # evaluation
            self.model.eval()
            val_losses = {
                'bce': [],
                'dice': [],
                'ct': []
            }
            val_ious = {name: [] for name in self.class_names}
            for i, (images, masks) in enumerate(ptable(self.val_loader, total=len(self.val_loader), description="Evaluation phase")):
                images = images.to(train_config['device'])
                masks = masks.to(train_config['device'])
                with torch.no_grad():
                    outputs, feat = self.model(images)
                losses = self.get_losses(outputs, masks, feat, outputs, masks, feat, contrast=False)
                torch.nn.CrossEntropyLoss()
                # metrics
                iou_all = self.metrics(outputs, masks)
                # log
                val_losses['bce'].append(losses['CEFocalLoss'].item())
                val_losses['dice'].append(losses['DiceLoss'].item())
                val_losses['ct'].append(losses['PixelWiseLabel'].item())

                ptable.update('Val BCE', np.mean(val_losses['bce']), aggregate="mean", color="yellow")
                ptable.update('Val Dice', np.mean(val_losses['dice']), aggregate="mean", color="cyan")
                ptable.update('Val CT', np.mean(val_losses['ct']), aggregate="mean", color="green")

                for c, iou in enumerate(iou_all):
                    val_ious[self.class_names[c]].append(iou.item())
                    ptable.update(self.class_names[c], np.mean(val_ious[self.class_names[c]]), aggregate="mean", color="magenta")
                
                iou_all_classes = np.mean([np.mean(v) for k, v in val_ious.items()])
                ptable.update("Val IoU", iou_all_classes, aggregate="mean", color="blue")
            
            self.writer.add_scalar("Val_Loss/BCE", np.mean(val_losses['bce']), epoch)
            self.writer.add_scalar("Val_Loss/Dice", np.mean(val_losses['dice']), epoch)
            for c, iou in enumerate(iou_all):
                val_ious[self.class_names[c]].append(iou.item())
                self.writer.add_scalar(f"Val_IoU/{c}", np.mean(val_ious[self.class_names[c]]), epoch)

            # save best model
            if iou_all_classes > best_iou:
                best_iou = iou_all_classes
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epochs': epoch
                }, f'{self.save_model_dir}/checkpoint.pth')
                torch.save(self.model.state_dict(), f'{self.save_model_dir}/best.pth')

            ptable.next_row(split=1)

        ptable.close()
    
if __name__ == '__main__':
    trainer = Trainer('config.yaml')
    trainer.train()