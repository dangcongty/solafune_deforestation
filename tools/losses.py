import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from shapely.geometry import Polygon
import time
import torch.nn.functional as F
from tqdm import tqdm

    
class CEFocalLoss(nn.Module):
    def __init__(self, weights = None, num_classes = 5, gamma = 2, reduction = 'mean'):
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, outputs: torch.Tensor, targets:torch.Tensor):
        '''
        Focal Loss: -(1-pt)^y * log(pt)
        pt = o*t - (1-o)*(1-t) 
        NOTE: log(1) = 0

        input:
            outputs: predict mask shape B x C x H x W 
            targets: target mask shape B x H x W 
        return:
            diceloss = 1 - (outputs ∩ targets)/(outputs + targets)
        '''
        bs, h, w = targets.shape
        outputs = outputs.log_softmax(1) # more numerically than softmax
        targets = F.one_hot(targets, self.num_classes) # BxHxW => BxHxWxC
        targets = targets.permute(0, -1, 1, 2).float() # BxHxWxC => BxCxHxW

        ce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
        p_t = outputs * targets + (1 - outputs) * (1 - targets)

        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.weights is not None:
            weights = self.weights.reshape((1, -1, 1, 1)).to(targets.device)
            loss *= weights

        if self.reduction == 'sum':
            return loss.sum()

        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes = 5, reduction = 'mean', weight = None, eps = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.eps = eps
        self.weight = weight

    def forward(self, outputs: torch.Tensor, targets:torch.Tensor):
        '''
        input:
            outputs: predict mask shape B x C x H x W 
            targets: target mask shape B x H x W 
        return:
            diceloss = 1 - (outputs ∩ targets)/(outputs + targets)
        '''
        bs, h, w = targets.shape

        outputs = outputs.softmax(1) # xác suất 
        
        targets = F.one_hot(targets, self.num_classes) # BxHxW => BxHxWxC
        targets = targets.permute(0, -1, 1, 2) # BxHxWxC => BxCxHxW

        intersection = (outputs * targets).sum(dim=(0, 2, 3))        
        union = outputs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        
        dice_score = (2. * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice_score
        if self.weight is not None:
            dice_loss = self.weight * dice_loss
        
        mask = targets.sum((0, 2, 3)) > 0 # Cho các trường hợp ko có class trong ảnh => dice = 0 => mean sẽ ko hợp lý
        dice_loss *= mask.to(dice_loss.dtype)
        
        if self.reduction == 'sum':
            return dice_loss.sum()
        
        return dice_loss.mean()

class PixelWiseLabel(nn.Module):
    def __init__(self, margin = 1, feature_size = 64):
        super().__init__()
        self.margin = margin
        self.feature_size = feature_size

    def get_simloss(self, o1, o2, t1, t2):
        cosine_sim = F.cosine_similarity(o1.unsqueeze(2), o2.unsqueeze(1), dim=0)  # Shape: (H*W, H*W)
        t_equal = (t1.unsqueeze(1) == t2.unsqueeze(0)).float()  # Shape: (H*W, H*W)
        t_unequal = 1 - t_equal
        cosine_sim = (cosine_sim + 1)/2 # shift & scale [-1, 1] => [0, 1]
        pos_loss = -torch.log(cosine_sim + 1e-8) * t_equal
        pos_loss = pos_loss.sum()/t_equal.sum() # mean
        neg_loss = -torch.log((1+1e-6)-cosine_sim + 1e-8) * t_unequal # 1+1e-6 for numerically
        neg_loss = neg_loss.sum()/t_unequal.sum() # mean
        total_loss = (pos_loss + neg_loss)/2
        return total_loss

    def forward(self, features1: torch.Tensor, targets1:torch.Tensor, features2: torch.Tensor, targets2:torch.Tensor, contrast = True):
        '''
        input:
            feature: predict mask shape B x C x H x W 
            targets: target mask shape B x H x W 
        '''

        # Resize feature map and GT mask
        features1 = F.interpolate(features1, size = (self.feature_size, self.feature_size), mode='bilinear', align_corners=False)
        targets1 = F.interpolate(targets1.float().unsqueeze(1), size = (self.feature_size, self.feature_size), mode='nearest').squeeze(1)
        if contrast:
            targets2 = F.interpolate(targets2.float().unsqueeze(1), size = (self.feature_size, self.feature_size), mode='nearest').squeeze(1)
            features2 = F.interpolate(features2, size = (self.feature_size, self.feature_size), mode='bilinear', align_corners=False)

        N, H, W = targets1.shape 
        N, C, H, W  = features1.shape

        loss_batches = []
        t1_flat = targets1.view(N, -1)  # Shape: (N, H*W)
        o1_flat = features1.view(N, C, -1)  # Shape: (N, C, H*W)
        if contrast:
            t2_flat = targets2.view(N, -1)  # Shape: (N, H*W)
            o2_flat = features2.view(N, C, -1)  # Shape: (N, C, H*W)

        for batch in range(N):
            t1 = t1_flat[batch]  # Shape: (H*W,)
            o1 = o1_flat[batch]  # Shape: (C, H*W)
            if contrast:
                t2 = t2_flat[batch]  # Shape: (H*W,)
                o2 = o2_flat[batch]  # Shape: (C, H*W)

            loss_within_image1 = self.get_simloss(o1, o1, t1, t1)
            if contrast:
                loss_within_image2 = self.get_simloss(o2, o2, t2, t2)
                loss_cross_image = self.get_simloss(o1, o2, t1, t2)
                total_loss = (loss_within_image1 + loss_within_image2 + loss_cross_image)/3
            else:
                total_loss = loss_within_image1
            loss_batches.append(total_loss)

        return torch.stack(loss_batches).mean()

class OhemCELoss(nn.Module):
    def __init__(self, thresh, device = 'cuda:1', lb_ignore=255):
        # https://github.dev/CoinCheung/BiSeNet/blob/master/lib/ohem_ce_loss.py
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).to(device)
        self.lb_ignore = lb_ignore
        self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.lb_ignore].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

if __name__ == '__main__':
    loss = PixelWiseLabel()
    outputs = torch.load('out.pt', weights_only=True)
    targets = torch.load('gt.pt', weights_only=True)
    loss(outputs, targets)