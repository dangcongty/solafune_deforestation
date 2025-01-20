import numpy as np
from base import BaseTrainer, set_seed
import torch
import torch.nn as nn
import cv2
import os
from progress_table import ProgressTable
from tools import OhemCELoss, Log
import torch.amp as amp

class Trainer(BaseTrainer):
    def __init__(self, config_file='config.yaml'):
        super().__init__(config_file)
        self.weight_cls = torch.from_numpy(np.load('dataset/weight_classes.npy')).to(self.config['Train']['device']).to(torch.float)
        self.__init_losses()

    def __init_losses(self, lb_ignore = 255):
        self.criteria_pre = OhemCELoss(0.7, lb_ignore=lb_ignore)
        self.criteria_aux = [OhemCELoss(0.7, lb_ignore=lb_ignore)
            for _ in range(4)]

    def get_losses(self, outputs, masks, stage = 'train'):
        losses = []
        for i, (o, m) in enumerate(zip(outputs, masks)):
            loss = torch.nn.CrossEntropyLoss(weight=self.weight_cls, reduction='sum')(o, m)
            losses.append(loss)
            if stage == 'train':
                self.train_losses[i].append(loss.item())
            else:
                self.val_losses[i].append(loss.item())

        loss_sum = torch.stack([*losses]).mean()
        if stage == 'train':
            self.train_losses[-1].append(loss_sum.item())
        else:
            self.val_losses[-1].append(loss_sum.item())
        return loss_sum, losses
    
    def find_iou(self, outputs, masks, stage = 'train'):
        outputs = outputs.softmax(1)
        iou = self.metrics(outputs, masks, threshold=0.5)
        for k in range(5):
            if stage == 'train':
                self.train_ious[k].append(iou[k].item())
            else:
                self.val_ious[k].append(iou[k].item())

    def train(self):

        train_config = self.config['Train']
        self.model = self.model.to(train_config['device'])
        ptable = ProgressTable(pbar_embedded=False, pbar_style="angled alt red blue")
        best_iou = 0
        scaler = amp.GradScaler()
        for epoch in range(1, train_config['epochs']+1):
            self.train_losses = [Log() for _ in range(len(self.config['Loader']['down'])+1)]
            self.train_ious = [Log() for _ in range(5)]

            self.val_losses = [Log() for _ in range(len(self.config['Loader']['down'])+1)]
            self.val_ious = [Log() for _ in range(5)]
            # train
            self.model.train()    
            ptable.update('Epoch          ', f'Train {epoch}/{train_config["epochs"]}', color="green")
            for i, ((images, masks)) in enumerate(ptable(self.train_loader, total=len(self.train_loader), description="Training phase")):
                images = images.to(train_config['device'])
                masks = masks.to(train_config['device'])
                with amp.autocast(device_type=train_config['device'], dtype=torch.float16):
                    logits, logits_aux = self.model(images)
                    loss_pre = self.criteria_pre(logits, masks)
                    loss_aux = [crit(lgt, masks) for crit, lgt in zip(self.criteria_aux, logits_aux)]
                    loss = loss_pre + sum(loss_aux)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                self.train_losses[-1].append(loss.item())
                self.train_losses[-2].append(loss_pre.item())
                _ = [self.train_losses[k].append(lss.item()) for k, lss in enumerate(loss_aux)]

                for k, d in enumerate(self.config['Loader']['down'][::-1] + ['total']):
                    ptable.update(f'Loss_{d}', self.train_losses[k].avg(), aggregate="mean", color="green")

                self.find_iou(logits, masks)
                # progress bar
                iou = 0
                for k in range(6):
                    if k < 5:
                        if k > 0:
                            iou += self.train_ious[k].avg()
                        ptable.update(f'IoU_{self.class_names[k]}', self.train_ious[k].avg(), aggregate="mean", color="green")
                    else:
                        ptable.update(f'IoU_total', iou/4, aggregate="mean", color="green")
            # tensorboard
            for k, d in enumerate(self.config['Loader']['down'][::-1] + ['total']):
                self.writer.add_scalar(f'Train/Loss_{d}', self.train_losses[k].avg(), epoch)
            iou = 0
            for k in range(6):
                if k < 5:
                    if k > 0:
                        iou += self.train_ious[k].avg()
                    self.writer.add_scalar(f'Train/IoU_{self.class_names[k]}', self.train_ious[k].avg(), epoch)
                else:
                    self.writer.add_scalar(f'Train/IoU_total', iou/4, epoch)


            # validation
            ptable.next_row(split=1)
            ptable.update('Epoch          ', f'Val {epoch}/{train_config["epochs"]}', color="green")
            self.model.eval()
            for i, (images, masks) in enumerate(ptable(self.val_loader, total=len(self.val_loader), description="Evaluation phase")):
                images = images.to(train_config['device'])
                masks = masks.to(train_config['device'])
                with torch.no_grad():
                    logits, logits_aux = self.model(images)
                    loss_pre = self.criteria_pre(logits, masks)


                self.val_losses[-1].append(loss.item())
                self.val_losses[-2].append(loss_pre.item())
                _ = [self.val_losses[k].append(lss.item()) for k, lss in enumerate(loss_aux)]

                for k, d in enumerate(self.config['Loader']['down'][::-1] + ['total']):
                    ptable.update(f'Loss_{d}', self.val_losses[k].avg(), aggregate="mean", color="green")

                self.find_iou(logits, masks, 'val')
                iou = 0
                for k in range(6):
                    if k < 5:
                        if k > 0:
                            iou += self.val_ious[k].avg()
                        ptable.update(f'IoU_{self.class_names[k]}', self.val_ious[k].avg(), aggregate="mean", color="green")
                    else:
                        ptable.update(f'IoU_total', iou/4, aggregate="mean", color="green")

            for k, d in enumerate(self.config['Loader']['down'][::-1] + ['total']):
                self.writer.add_scalar(f'Val/Loss_{d}', self.val_losses[k].avg(), epoch)            
            iou = 0
            for k in range(6):
                if k < 5:
                    if k > 0:
                        iou += self.val_ious[k].avg()
                    self.writer.add_scalar(f'Val/IoU_{self.class_names[k]}', self.val_ious[k].avg(), epoch)
                else:
                    self.writer.add_scalar(f'Val/IoU_total', iou/4, epoch)
                    
            iou_all_classes = np.mean([self.val_ious[k].avg() for k in range(5)][1:])
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



trainer = Trainer()
trainer.train()