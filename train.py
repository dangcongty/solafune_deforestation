import os
import shutil
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.amp as amp
import torch.optim as optim
import yaml
from progress_table import ProgressTable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from base import set_seed
from models.yolo import YOLOSeg
from tools import Log, OhemCELoss, seed_everything
from tools.dataset import Data
from tools.metrics import compute_iou


class Trainer:
    def __init__(self, config_file='config.yaml'):
        super().__init__()

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        train_config = self.config['Train']
        model_config = self.config['Model']
        loader_config = self.config['Loader']
        optim_config = self.config['Optimizer']

        # Initialize model
        self.model = YOLOSeg(config_file=model_config['config_file'], scale=model_config['scale']).to(train_config['device'])

        # Initialize dataset loaders
        self.train_loader = self._create_dataloader(mode='train', batch_size=loader_config['train_bs'])
        self.val_loader = self._create_dataloader(mode='val', batch_size=loader_config['val_bs'])

        # Initialize optimizer
        optimizer_class = getattr(optim, optim_config["type"], None)
        self.optimizer = optimizer_class(self.model.parameters(), lr=optim_config['lr'])

        # Initialize loss functions
        device = train_config['device']
        self.criteria_pre = OhemCELoss(0.7, device=device, lb_ignore=255)
        self.criteria_aux = [OhemCELoss(0.7, device=device, lb_ignore=255) for _ in range(4)]
        self.weight_cls = torch.from_numpy(np.load('dataset/weight_classes.npy')).to(device).float()

        # Initialize workspace
        self.save_model_dir = self._init_workspace()

        # Class names
        # self.class_names = ['background', "grassland_shrubland", "logging", "mining", "plantation"]
        self.class_names = ['background', "logging"]

    def _create_dataloader(self, mode, batch_size):
        """Creates a DataLoader for the given mode (train/val)."""
        dataset = Data(mode=mode, config=self.config)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'),
                          drop_last=(mode == 'train'), pin_memory=True, num_workers=2)

    def _init_workspace(self):
        """Initializes workspace for saving models and logs."""
        os.makedirs('ckpt', exist_ok=True)
        version = len(os.listdir('ckpt')) + 1
        save_dir = f'ckpt/v{version}'
        os.makedirs(save_dir, exist_ok=True)

        shutil.copyfile('config.yaml', f'{save_dir}/config.yaml')
        shutil.copyfile('train.py', f'{save_dir}/train.py')
        shutil.copytree('tools', f'{save_dir}/tools')
        shutil.copytree('models', f'{save_dir}/models')

        self.writer = SummaryWriter(log_dir=f'{save_dir}/log')
        return save_dir

    def find_iou(self, outputs, masks, stage='train'):
        """Computes IoU and stores results."""
        outputs = outputs.softmax(1)
        ious, iou_mask = compute_iou(outputs, masks, threshold=0.5)
        # if iou_mask[1] != 0:
        iou_value = ious[1].item()
        if stage == 'train':
            self.train_ious[1].append(iou_value)
        else:
            self.val_ious[1].append(iou_value)

    def train(self):
        """Trains the YOLO segmentation model."""
        train_config = self.config['Train']
        epochs = train_config['epochs']
        ptable = ProgressTable(pbar_embedded=False, pbar_style="angled alt red blue")
        scaler = amp.GradScaler()
        best_iou = 0

        for epoch in range(1, epochs + 1):
            self._reset_logs()

            # Training Phase
            self.model.train()
            pbar = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Training phase: {epoch}", bar_format='{l_bar}{bar:10}{r_bar}')
            for z, (images, masks) in enumerate(pbar):
                images, masks = images.to(train_config['device']), masks.to(train_config['device'])
                t2 = time.time()
                with amp.autocast(device_type=train_config['device'], dtype=torch.float16):
                    logits, logits_aux, _ = self.model(images)
                    loss_pre = self.criteria_pre(logits, masks)
                    loss_aux = sum(crit(lgt, masks) for crit, lgt in zip(self.criteria_aux, logits_aux))
                    loss = (loss_pre + loss_aux) / 5
                t3 = time.time()
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                t4 = time.time()
                self._log_losses(loss, loss_pre, loss_aux, stage="train")
                self.find_iou(logits, masks, 'train')
                t5 = time.time()

                if z % 100 == 0:
                    # write result
                    pred = logits.argmax(1)
                    v = np.random.randint(0, len(pred)-1)
                    vis_pred = pred[v].cpu().numpy()*2555
                    vis_gt = masks[v].cpu().numpy()*2555
                    vis = np.concatenate([vis_pred, np.zeros((1024, 50)), vis_gt], 1)
                    cv2.imwrite('vis.jpg', vis)

                # print(f'data: {t2-t1:.4f} model+loss: {t3-t2:.4f} update: {t4-t3:.4f} iou+log: {t5-t4:.4f}')
            self._log_tensorboard(epoch, stage="train")

            # Validation Phase
            self.model.eval()
            with torch.no_grad():
                for images, masks in tqdm(self.val_loader, total=len(self.val_loader), desc=f"Evaluation phase: {epoch}", bar_format='{l_bar}{bar:10}{r_bar}'):
                    images, masks = images.to(train_config['device']), masks.to(train_config['device'])
                    logits, logits_aux, _ = self.model(images)

                    loss_pre = self.criteria_pre(logits, masks)
                    loss_aux = sum(crit(lgt, masks) for crit, lgt in zip(self.criteria_aux, logits_aux))
                    loss = (loss_pre + loss_aux) / 5

                    self._log_losses(loss, loss_pre, loss_aux, stage="val")
                    self.find_iou(logits, masks, 'val')

            self._log_tensorboard(epoch, stage="val")

            # Save Best Model
            mean_iou = self.val_ious[1].avg()
            if mean_iou > best_iou:
                best_iou = mean_iou
                self._save_checkpoint(epoch)


    def _reset_logs(self):
        """Resets loss and IoU logs for each epoch."""
        self.train_losses = defaultdict(Log)
        self.train_ious = defaultdict(Log)
        self.val_losses = defaultdict(Log)
        self.val_ious = defaultdict(Log)

    def _log_losses(self, loss, loss_pre, loss_aux, stage):
        """Stores loss values for logging."""
        losses = self.train_losses if stage == "train" else self.val_losses
        losses['total'].append(loss.item())
        losses['pre'].append(loss_pre.item())
        losses['aux'].append(loss_aux.item())

    def _log_tensorboard(self, epoch, stage):
        """Logs loss and IoU values to TensorBoard."""
        losses = self.train_losses if stage == "train" else self.val_losses
        ious = self.train_ious if stage == "train" else self.val_ious

        # Log losses
        for key, log in losses.items():
            self.writer.add_scalar(f'{stage.capitalize()}/Loss_{key}', log.avg(), epoch)

        # Log IoU per class
        mean_iou = 0
        class_iou = ious[1].avg()
        mean_iou += class_iou
        self.writer.add_scalar(f'{stage.capitalize()}/IoU_{self.class_names[1]}', class_iou, epoch)

        # Log total mean IoU (excluding background)
        self.writer.add_scalar(f'{stage.capitalize()}/IoU_total', mean_iou / 4, epoch)

    def _save_checkpoint(self, epoch):
        """Saves model checkpoints."""
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'epochs': epoch},
                   f'{self.save_model_dir}/checkpoint.pth')
        torch.save(self.model.state_dict(), f'{self.save_model_dir}/best.pth')


if __name__ == '__main__':
    seed_everything()
    trainer = Trainer()
    trainer.train()
