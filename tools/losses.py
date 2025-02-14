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
    def __init__(self):
        super().__init__()

    def cosine_similarity(self, f1, f2):
        dot_product = torch.matmul(f1, f2.T)
        norms1 = torch.norm(f1, dim=1, keepdim=True)  # Keepdim=True for broadcasting
        norms2 = torch.norm(f2, dim=1, keepdim=True)
        norm_matrix = norms1 * norms2.T  # Equivalent to outer product
        cosine_sim = dot_product / (norm_matrix + 1e-8)
        return  cosine_sim.mean()

    def forward(self, features: torch.Tensor, targets:torch.Tensor):
        '''
        input:
            feature: predict mask shape B x 32 x 256 x 256 
            targets: target mask shape B x 1024 x 1024
        '''
        losses = []
        ignore = -1
        H, W = features.shape[2:]
        targets = F.interpolate(targets.unsqueeze(1).float(), (H, W), mode='nearest').squeeze(1)
        # 1 images - multiple classes
        for b1 in range(targets.shape[0]):
            ignore += 1
            for b2 in range(ignore, targets.shape[0]):
                for c in range(5):
                    if b1 == b2:
                        # pixel level
                        t = (targets[b1] == c)*1
                        mask = torch.nonzero(t.flatten(), as_tuple=False).squeeze(1)
                        if not len(mask):
                            continue
                        # t_mask = t.flatten()[mask]
                        f = features[b1].flatten(1, 2)
                        f_mask = f[:, mask].permute((1, 0))

                        cosine_score = []
                        bsize = 2048
                        for idx1 in range(0, len(f_mask), bsize):
                            fbatch1 = f_mask[idx1: idx1+bsize]
                            for idx2 in range(0, len(f_mask), bsize):
                                fbatch2 = f_mask[idx2: idx2+bsize]
                                sim = self.cosine_similarity(fbatch1, fbatch2)
                                cosine_score.append(sim)
                        cosine_score = torch.stack(cosine_score).mean()
                        cosine_score = (cosine_score+1)/2 # shift & scale
                        loss = -torch.log(cosine_score + 1e-4)

                    else:
                        # cross image level
                        t1 = (targets[b1] == c)*1
                        mask1 = torch.nonzero(t1.flatten(), as_tuple=False).squeeze(1)
                        if not len(mask1):
                            continue
                        f1 = features[b1].flatten(1, 2)
                        f1_mask = f1[:, mask1].permute((1, 0))

                        t2 = (targets[b2] == c)*1
                        mask2 = torch.nonzero(t2.flatten(), as_tuple=False).squeeze(1)
                        f2 = features[b2].flatten(1, 2)
                        f2_mask = f2[:, mask2].permute((1, 0))
                        if not len(mask2):
                            continue
                        cosine_score = []
                        bsize = 2048
                        for idx1 in range(0, len(f1_mask), bsize):
                            fbatch1 = f1_mask[idx1: idx1+bsize]
                            for idx2 in range(0, len(f2_mask), bsize):
                                fbatch2 = f2_mask[idx2: idx2+bsize]
                                sim = self.cosine_similarity(fbatch1, fbatch2)
                                cosine_score.append(sim)
                        cosine_score = torch.stack(cosine_score).mean()
                        cosine_score = (cosine_score+1)/2
                        loss = -torch.log(cosine_score + 1e-4)

                    losses.append(loss)

        return torch.stack(losses).mean()

class OhemCELoss(nn.Module):
    def __init__(self, thresh, device = 'cuda:1', lb_ignore=255):
        # https://github.dev/CoinCheung/BiSeNet/blob/master/lib/ohem_ce_loss.py
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).to(device)
        self.lb_ignore = lb_ignore
        self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.lb_ignore].numel() // 16
        loss = self.criteria(logits, labels.to(torch.long)).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

if __name__ == '__main__':
    loss = PixelWiseLabel()
    outputs = torch.rand((4, 128, 128, 128)).cuda()
    outputs[0] -= 0.5
    targets = torch.randint(0, 5, (4, 128, 128)).cuda()
    loss(outputs, targets, outputs, targets)